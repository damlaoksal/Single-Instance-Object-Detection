#include "YOLODetector.h"
#include <iostream>
#include <algorithm>
#include <cmath>

YOLODetector::YOLODetector()
    : m_env(ORT_LOGGING_LEVEL_WARNING, "YOLODetector")
    , m_session(nullptr)
    , m_memoryInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
{
}

bool YOLODetector::initialize(const std::string& modelPath, float confThreshold, float nmsThreshold)
{
    m_confThreshold = confThreshold;
    m_nmsThreshold = nmsThreshold;

    try {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(2);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        try {
            OrtCUDAProviderOptions cudaOptions;
            cudaOptions.device_id = 0;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
            std::cout << "[YOLO] CUDA GPU Enabled!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "[YOLO WARNING] CUDA failed, falling back to CPU: " << e.what() << std::endl;
        }

#ifdef _WIN32
        std::wstring wModelPath(modelPath.begin(), modelPath.end());
        m_session = Ort::Session(m_env, wModelPath.c_str(), sessionOptions);
#else
        m_session = Ort::Session(m_env, modelPath.c_str(), sessionOptions);
#endif

        Ort::AllocatorWithDefaultOptions allocator;

        if (m_session.GetInputCount() > 0) {
            m_inputName = m_session.GetInputNameAllocated(0, allocator).get();
            m_inputNames.push_back(m_inputName.c_str());
            m_inputShape = m_session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        }

        if (m_session.GetOutputCount() > 0) {
            m_outputName = m_session.GetOutputNameAllocated(0, allocator).get();
            m_outputNames.push_back(m_outputName.c_str());
        }

        cv::Mat dummyMat = cv::Mat::zeros(640, 640, CV_8UC3);
        detect(dummyMat);

        std::cout << "[YOLO] Model Ready! (Conf: " << m_confThreshold << ")" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[YOLO ERROR] Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat YOLODetector::preprocess(const cv::Mat& frame, float& scaleX, float& scaleY)
{
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(m_inputWidth, m_inputHeight));

    scaleX = static_cast<float>(frame.cols) / static_cast<float>(m_inputWidth);
    scaleY = static_cast<float>(frame.rows) / static_cast<float>(m_inputHeight);

    cv::Mat rgb, floatMat;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(floatMat, CV_32FC3, 1.0 / 255.0);

    return floatMat;
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& frame)
{
    std::vector<Detection> detections;
    if (frame.empty() || !m_session) return detections;

    float scaleX = 1.0f, scaleY = 1.0f;
    cv::Mat preprocessedMat = preprocess(frame, scaleX, scaleY);

    std::vector<float> inputTensorValues(1 * 3 * m_inputHeight * m_inputWidth);
    int channelSize = m_inputHeight * m_inputWidth;

    //Pointer-based fast copy (Reduces CPU load for Edge devices)
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; ++i) {
        channels[i] = cv::Mat(m_inputHeight, m_inputWidth, CV_32FC1, inputTensorValues.data() + i * channelSize);
    }
    cv::split(preprocessedMat, channels);

    std::vector<int64_t> inputShape = { 1, 3, m_inputHeight, m_inputWidth };
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        m_memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShape.data(), inputShape.size()
    );

    try {
        auto outputTensors = m_session.Run(
            Ort::RunOptions{ nullptr },
            m_inputNames.data(), &inputTensor, 1,
            m_outputNames.data(), 1
        );

        float* floatArray = outputTensors[0].GetTensorMutableData<float>();
        auto shape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        int numAttributes = static_cast<int>(shape[1]);
        int numBoxes = static_cast<int>(shape[2]);

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;

        for (int i = 0; i < numBoxes; ++i) {
            float cx = floatArray[0 * numBoxes + i];
            float cy = floatArray[1 * numBoxes + i];
            float w  = floatArray[2 * numBoxes + i];
            float h  = floatArray[3 * numBoxes + i];

            int maxClassId = -1;
            float maxScore = 0.0f;

            for (int c = 4; c < numAttributes; ++c) {
                float score = floatArray[c * numBoxes + i];
                if (score > maxScore) {
                    maxScore = score;
                    maxClassId = c - 4;
                }
            }

            bool isPerson = (maxClassId == 0);
            bool isVehicle = (maxClassId == 2 || maxClassId == 3 || maxClassId == 5 || maxClassId == 7);

            if (maxScore >= m_confThreshold && (isPerson || isVehicle)) {
                int left = static_cast<int>((cx - 0.5f * w) * scaleX);
                int top  = static_cast<int>((cy - 0.5f * h) * scaleY);
                int width = static_cast<int>(w * scaleX);
                int height = static_cast<int>(h * scaleY);

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(maxScore);
                classIds.push_back(isPerson ? 0 : 2);
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices);

        for (int idx : indices) {
            Detection det;
            det.box = boxes[idx] & cv::Rect(0, 0, frame.cols, frame.rows);
            det.confidence = confidences[idx];
            det.classId = classIds[idx];
            detections.push_back(det);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[YOLO ERROR] Inference error: " << e.what() << std::endl;
    }

    return detections;
}