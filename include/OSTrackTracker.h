#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <filesystem>

// Stores the tracking outcome for the current frame
struct TrackResult {
    cv::Rect box;          // Bounding box of the tracked object in pixel coordinates
    float confidence = 0.0f; // Confidence score of the detection/tracking
};

class OSTrackTracker {
public:
    OSTrackTracker();
    ~OSTrackTracker();

    // Loads the ONNX model and sets up the ONNX Runtime environment
    bool initialize(const std::string& modelPath);
    
    // Initializes the tracker with the first frame and the initial bounding box of the target
    void init(const cv::Mat& frame, const cv::Rect& bbox);
    
    // Performs tracking on a new frame and returns the updated bounding box
    TrackResult update(const cv::Mat& frame);

private:
    // Crops and resizes the region of interest centered around the target for the tracker backbone
    cv::Mat cropAndResize(const cv::Mat& img, cv::Point2f center, float size, int outputSize);
    
    // Preprocesses the image frame (normalization, layout conversion) directly into the preallocated tensor buffer
    void preprocessImagePreallocated(const cv::Mat& img, float* outputBuffer);

    // ONNX Runtime core components
    Ort::Env m_env;
    Ort::SessionOptions m_sessionOptions;
    Ort::Session* m_session = nullptr;
    Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Network input dimensions expected by OSTrack
    const int m_templateSize = 128;
    const int m_searchSize = 256;

    // Preallocated tensor buffers to avoid frequent memory reallocations during inference
    std::vector<float> m_templateTensorData;
    std::vector<float> m_searchTensorData;

    // Internal state variables for tracking the target across frames
    cv::Point2f m_targetCenter;
    cv::Size2f m_targetSize;
    cv::Mat m_templateCrop;
    bool m_isInitialized = false;

    // Model I/O node names used for ONNX Runtime inference execution
    std::vector<const char*> m_inputNames = {"template", "search"};
    std::vector<const char*> m_outputNames = {"output"};
};