#pragma once

#include <string>
#include <vector>
#include <deque>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "YOLODetector.h"
#include "OSTrackTracker.h"
#include "Detection.h"

enum class SystemState {
    DETECTION,     
    TRACKING,      
    RE_DETECTION   
};

class VideoPlayer {
public:
    explicit VideoPlayer(const std::string& videoPath, bool enableGUI = true);
    ~VideoPlayer() = default;

    bool initialize();
    void run();
    void setTargetTemplate(const std::string& imagePath);

private:
    void processFrame(cv::Mat& frame);
    static void onMouse(int event, int x, int y, int flags, void* userdata);

    std::string m_videoPath;
    std::string m_windowName = "Operator-Assisted Template Radar Engine";
    cv::VideoCapture m_capture;
    cv::dnn::Net m_osnetNet;

    YOLODetector m_detector;
    OSTrackTracker m_tracker;

    SystemState m_currentState = SystemState::DETECTION;
    bool m_trackerInitialized = false;

    cv::Mat m_targetTemplateImage;
    bool m_hasTemplateImage = false;

    bool m_isDragging = false;
    cv::Point m_dragStartPoint;
    cv::Rect m_userDefinedRoi = cv::Rect(0, 0, 0, 0);
    bool m_hasUserRoi = false;

    cv::Rect m_selectedTarget;
    cv::Point2f m_lastKnownCenter;
    cv::Point2f m_velocity = cv::Point2f(0.0f, 0.0f);
    int m_occlusionFrameCounter = 0;
    int m_gracePeriodCounter = 0;

    int m_targetClassId = -1;                            
    std::string m_targetClassName = "";                         

    float m_targetAspectRatio = 1.0f;                         
    float m_targetInitialArea = 0.0f;

    bool m_enableGUI = true;                         
    cv::Mat m_displayFrame;                            
    double m_originalFPS = 30.0;                           
    double m_currentFPS = 0.0;                             
    double m_processLatencyMs = 0.0;                            
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;

    std::vector<Detection> m_currentDetections;
    std::deque<std::vector<Detection>> m_detectionHistory;
    const size_t m_maxHistoryFrames = 5;
};