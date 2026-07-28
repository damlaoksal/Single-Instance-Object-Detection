#include "OSTrackTracker.h"
#include <iostream>
#include <cmath>
#include <algorithm>

OSTrackTracker::OSTrackTracker()
    : m_env(ORT_LOGGING_LEVEL_WARNING, "OSTrackTrackerGPU")
{
    m_templateTensorData.resize(1 * 3 * m_templateSize * m_templateSize);
    m_searchTensorData.resize(1 * 3 * m_searchSize * m_searchSize);
}

OSTrackTracker::~OSTrackTracker()
{
    delete m_session;
    m_session = nullptr;
}

bool OSTrackTracker::initialize(const std::string& modelPath)
{
    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "[OSTrack ERROR] Model file not found: " << modelPath << std::endl;
        return false;
    }

    try {
        OrtCUDAProviderOptions cudaOptions;
        cudaOptions.device_id = 0;
        cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cudaOptions.gpu_mem_limit = SIZE_MAX;
        cudaOptions.arena_extend_strategy = 0;

        m_sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        m_sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        m_sessionOptions.SetIntraOpNumThreads(2);

#if defined(_WIN32)
        std::wstring wModelPath(modelPath.begin(), modelPath.end());
        m_session = new Ort::Session(m_env, wModelPath.c_str(), m_sessionOptions);
#else
        m_session = new Ort::Session(m_env, modelPath.c_str(), m_sessionOptions);
#endif
        std::cout << "[OSTrack GPU] Model loaded successfully: " << modelPath << std::endl;
        return true;
    }
    catch (...) {
        std::cerr << "[OSTrack WARNING] CUDA failed, falling back to CPU." << std::endl;
        try {
            Ort::SessionOptions cpuOptions;
            cpuOptions.SetIntraOpNumThreads(4);
#if defined(_WIN32)
            std::wstring wModelPath(modelPath.begin(), modelPath.end());
            m_session = new Ort::Session(m_env, wModelPath.c_str(), cpuOptions);
#else
            m_session = new Ort::Session(m_env, modelPath.c_str(), cpuOptions);
#endif
            std::cout << "[OSTrack] Model loaded on CPU!" << std::endl;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "[OSTrack ERROR] CPU load failed: " << e.what() << std::endl;
            return false;
        }
    }
}

cv::Mat OSTrackTracker::cropAndResize(const cv::Mat& img, cv::Point2f center, float size, int outputSize)
{
    size = std::max(1.0f, size);
    float x1 = center.x - size / 2.0f;
    float y1 = center.y - size / 2.0f;

    int ix1 = static_cast<int>(std::floor(x1));
    int iy1 = static_cast<int>(std::floor(y1));
    int ix2 = ix1 + static_cast<int>(std::ceil(size));
    int iy2 = iy1 + static_cast<int>(std::ceil(size));

    int padX1 = std::max(0, -ix1);
    int padY1 = std::max(0, -iy1);
    int padX2 = std::max(0, ix2 - img.cols);
    int padY2 = std::max(0, iy2 - img.rows);

    cv::Rect roi(ix1 + padX1, iy1 + padY1, (ix2 - padX2) - (ix1 + padX1), (iy2 - padY2) - (iy1 + padY1));

    cv::Mat cropped;
    if (roi.width > 0 && roi.height > 0 && roi.x >= 0 && roi.y >= 0 &&
        (roi.x + roi.width) <= img.cols && (roi.y + roi.height) <= img.rows) {
        cv::copyMakeBorder(img(roi), cropped, padY1, padY2, padX1, padX2, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } else {
        cropped = cv::Mat::zeros(std::max(1, static_cast<int>(size)), std::max(1, static_cast<int>(size)), img.type());
    }

    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(outputSize, outputSize), 0, 0, cv::INTER_LINEAR);
    return resized;
}

void OSTrackTracker::preprocessImagePreallocated(const cv::Mat& img, float* outputBuffer)
{
    int channelSize = img.rows * img.cols;
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};

    float* rChannel = outputBuffer;
    float* gChannel = outputBuffer + channelSize;
    float* bChannel = outputBuffer + 2 * channelSize;

    for (int i = 0; i < channelSize; ++i) {
        const cv::Vec3b& pixel = img.at<cv::Vec3b>(i);
        bChannel[i] = ((pixel[0] / 255.0f) - mean[2]) / std[2];
        gChannel[i] = ((pixel[1] / 255.0f) - mean[1]) / std[1];
        rChannel[i] = ((pixel[2] / 255.0f) - mean[0]) / std[0];
    }
}

void OSTrackTracker::init(const cv::Mat& frame, const cv::Rect& bbox)
{
    m_targetCenter = cv::Point2f(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
    m_targetSize = cv::Size2f(static_cast<float>(bbox.width), static_cast<float>(bbox.height));

    float w = m_targetSize.width;
    float h = m_targetSize.height;
    float cropFactor = std::sqrt((w + 0.5f * (w + h)) * (h + 0.5f * (w + h))) * 2.0f;

    m_templateCrop = cropAndResize(frame, m_targetCenter, cropFactor, m_templateSize);
    preprocessImagePreallocated(m_templateCrop, m_templateTensorData.data());

    m_isInitialized = true;
    std::cout << "[OSTrack] Target initialized at: (" << m_targetCenter.x << ", " << m_targetCenter.y << ")" << std::endl;
}

TrackResult OSTrackTracker::update(const cv::Mat& frame)
{
    TrackResult result;
    if (!m_isInitialized || !m_session) return result;

    float w = m_targetSize.width;
    float h = m_targetSize.height;
    float cropFactor = std::sqrt((w + 0.5f * (w + h)) * (h + 0.5f * (w + h))) * 2.0f;

    cv::Mat searchCrop = cropAndResize(frame, m_targetCenter, cropFactor, m_searchSize);
    preprocessImagePreallocated(searchCrop, m_searchTensorData.data());

    // Allocate shapes on stack to prevent heap fragmentation (Edge optimization)
    const int64_t templateShape[] = {1, 3, m_templateSize, m_templateSize};
    const int64_t searchShape[] = {1, 3, m_searchSize, m_searchSize};

    // Use raw array instead of std::vector to avoid dynamic memory allocation per frame
    Ort::Value inputTensors[2] = {
        Ort::Value::CreateTensor<float>(m_memoryInfo, m_templateTensorData.data(), m_templateTensorData.size(), templateShape, 4),
        Ort::Value::CreateTensor<float>(m_memoryInfo, m_searchTensorData.data(), m_searchTensorData.size(), searchShape, 4)
    };

    auto outputTensors = m_session->Run(Ort::RunOptions{nullptr}, m_inputNames.data(), inputTensors, 2, m_outputNames.data(), 1);
    float* outputData = outputTensors[0].GetTensorMutableData<float>();

    float p0 = outputData[0], p1 = outputData[1], p2 = outputData[2], p3 = outputData[3];

    if (p2 > 1.0f || p3 > 1.0f) {
        p0 /= m_searchSize; p1 /= m_searchSize; p2 /= m_searchSize; p3 /= m_searchSize;
    }

    float predCenterX = (p2 > p0 && (p2 - p0) > 0.01f) ? (p0 + p2) / 2.0f : p0;
    float predCenterY = (p2 > p0 && (p2 - p0) > 0.01f) ? (p1 + p3) / 2.0f : p1;

    float offsetX = (predCenterX - 0.5f) * cropFactor;
    float offsetY = (predCenterY - 0.5f) * cropFactor;

    if (std::abs(offsetX) > 1.2f) m_targetCenter.x += offsetX * 0.85f;
    if (std::abs(offsetY) > 1.2f) m_targetCenter.y += offsetY * 0.85f;

    int resX = std::max(0, std::min(static_cast<int>(std::round(m_targetCenter.x - w / 2.0f)), frame.cols - 1));
    int resY = std::max(0, std::min(static_cast<int>(std::round(m_targetCenter.y - h / 2.0f)), frame.rows - 1));
    int resW = std::max(10, std::min(static_cast<int>(std::round(w)), frame.cols - resX));
    int resH = std::max(10, std::min(static_cast<int>(std::round(h)), frame.rows - resY));

    float maxDist = cropFactor * 0.5f;
    float dist = std::sqrt(offsetX * offsetX + offsetY * offsetY);
    
    result.box = cv::Rect(resX, resY, resW, resH);
    result.confidence = 1.0f - std::min(1.0f, dist / maxDist);

    return result;
}