#ifndef YOLODETECTOR_H
#define YOLODETECTOR_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "Detection.h"

// Handles YOLO object detection inference using ONNX Runtime.
// Make sure to initialize the model properly before running detections.
class YOLODetector {
public:
    YOLODetector();
    ~YOLODetector() = default;

    // Loads the ONNX model and sets up the inference session with given thresholds.
    bool initialize(const std::string& modelPath, float confThreshold = 0.35f, float nmsThreshold = 0.45f);
    
    // Runs inference on a single frame and returns a list of detected objects.
    std::vector<Detection> detect(const cv::Mat& frame);

private:
    // Resizes and normalizes the input frame to match model dimensions.
    cv::Mat preprocess(const cv::Mat& frame, float& scaleX, float& scaleY);

    Ort::Env m_env;
    Ort::Session m_session;
    Ort::MemoryInfo m_memoryInfo;

    std::string m_inputName;
    std::string m_outputName;
    std::vector<const char*> m_inputNames;
    std::vector<const char*> m_outputNames;
    std::vector<int64_t> m_inputShape;

    // Default input dimensions expected by the YOLO model
    const int m_inputWidth = 640;
    const int m_inputHeight = 640;
    
    float m_confThreshold = 0.35f;
    float m_nmsThreshold = 0.45f;
};

#endif // YOLODETECTOR_H