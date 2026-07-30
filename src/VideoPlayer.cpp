#include "VideoPlayer.h"
#include "ImageUtils.h"
#include "VehicleMatcher.h"
#include "PersonMatcher.h"
#include <iostream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

VideoPlayer::VideoPlayer(const std::string& videoPath, bool enableGUI)
    : m_videoPath(videoPath), m_enableGUI(enableGUI)
{
}

bool VideoPlayer::initialize()
{
    m_capture.open(m_videoPath);
    if (!m_capture.isOpened()) {
        std::cerr << "Error: Could not open video: " << m_videoPath << std::endl;
        return false;
    }

    double fps = m_capture.get(cv::CAP_PROP_FPS);
    if (fps > 0.0 && !std::isnan(fps)) m_originalFPS = fps;

    std::string yoloPath = std::filesystem::exists("models_onnx/yolo11s.onnx") ? "models_onnx/yolo11s.onnx" : "../../models_onnx/yolo11s.onnx";
    if (!m_detector.initialize(yoloPath, 0.1f,0.45f)) {
        std::cerr << "Error: Failed to load YOLO model!" << std::endl;
        return false;
    }

    std::string ostrackPath = std::filesystem::exists("models_onnx/ostrack_256.onnx") ? "models_onnx/ostrack_256.onnx" : "../../models_onnx/ostrack_256.onnx";
    if (!m_tracker.initialize(ostrackPath)) {
        std::cerr << "Error: Failed to load OSTrack model!" << std::endl;
        return false;
    }

    std::string osnetPath = std::filesystem::exists("models_onnx/osnet_ain_x0_5.onnx") ? "models_onnx/osnet_ain_x0_5.onnx" : "../../models_onnx/osnet_ain_x0_5.onnx";
    try {
        m_osnetNet = cv::dnn::readNetFromONNX(osnetPath);
        m_osnetNet.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        m_osnetNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } catch (const cv::Exception& e) {
        std::cerr << "Error: Failed to load OSNet model: " << e.what() << std::endl;
        return false;
    }

    std::cout << "[ENGINE] Hybrid System Ready! FPS: " << m_originalFPS << std::endl;
    return true;
}

void VideoPlayer::setTargetTemplate(const std::string& imagePath)
{
    m_targetTemplateImage = cv::imread(imagePath);
    if (!m_targetTemplateImage.empty()) {
        m_hasTemplateImage = true;
        std::cout << "[TEMPLATE] Target crop loaded: " << imagePath << std::endl;
    } else {
        std::cerr << "Error: Failed to read crop image: " << imagePath << std::endl;
    }
}

void VideoPlayer::onMouse(int event, int x, int y, int flags, void* userdata)
{
    VideoPlayer* self = static_cast<VideoPlayer*>(userdata);

    double origW = self->m_capture.get(cv::CAP_PROP_FRAME_WIDTH);
    double origH = self->m_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    if (origW <= 0 || origH <= 0) return;

    int realX = static_cast<int>(std::round(x * (origW / 1280.0)));
    int realY = static_cast<int>(std::round(y * (origH / 720.0)));

    if (self->m_currentState == SystemState::DETECTION && !self->m_hasTemplateImage) {
        if (event != cv::EVENT_LBUTTONDOWN) return;
        cv::Point clickPoint(realX, realY);

        for (auto it = self->m_detectionHistory.rbegin(); it != self->m_detectionHistory.rend(); ++it) {
            for (const auto& det : *it) {
                if (det.box.contains(clickPoint)) {
                    self->m_selectedTarget = det.box;
                    self->m_targetClassId = det.classId;
                    self->m_targetClassName = (det.classId == 0) ? "Person" : "Vehicle";
                    self->m_targetAspectRatio = static_cast<float>(det.box.width) / det.box.height;
                    self->m_targetInitialArea = static_cast<float>(det.box.width * det.box.height);
                    self->m_lastKnownCenter = cv::Point2f(det.box.x + det.box.width / 2.0f, det.box.y + det.box.height / 2.0f);
                    self->m_velocity = cv::Point2f(0.0f, 0.0f);
                    self->m_currentState = SystemState::TRACKING;
                    self->m_trackerInitialized = false;
                    std::cout << "[TARGET LOCKED]\n" << std::flush;
                    return;
                }
            }
        }
    } else if (self->m_currentState == SystemState::RE_DETECTION) {
        if (event == cv::EVENT_LBUTTONDOWN) {
            self->m_isDragging = true;
            self->m_dragStartPoint = cv::Point(realX, realY);
            self->m_hasUserRoi = false;
        } else if (event == cv::EVENT_MOUSEMOVE && self->m_isDragging) {
            int w = realX - self->m_dragStartPoint.x;
            int h = realY - self->m_dragStartPoint.y;
            self->m_userDefinedRoi = cv::Rect(w > 0 ? self->m_dragStartPoint.x : realX, h > 0 ? self->m_dragStartPoint.y : realY, std::abs(w), std::abs(h));
        } else if (event == cv::EVENT_LBUTTONUP && self->m_isDragging) {
            self->m_isDragging = false;
            int w = realX - self->m_dragStartPoint.x;
            int h = realY - self->m_dragStartPoint.y;
            if (std::abs(w) > 10 && std::abs(h) > 10) {
                self->m_userDefinedRoi = cv::Rect(w > 0 ? self->m_dragStartPoint.x : realX, h > 0 ? self->m_dragStartPoint.y : realY, std::abs(w), std::abs(h));
                self->m_hasUserRoi = true;
            }
        }
    }
}

void VideoPlayer::run()
{
    if (m_enableGUI) {
        cv::namedWindow(m_windowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(m_windowName, 1280, 720);
        cv::setMouseCallback(m_windowName, VideoPlayer::onMouse, this);
    }

    cv::Mat frame;
    m_lastFrameTime = std::chrono::high_resolution_clock::now();

    while (m_capture.read(frame)) {
        if (frame.empty()) break;

        auto startTime = std::chrono::high_resolution_clock::now();
        processFrame(frame);
        auto endTime = std::chrono::high_resolution_clock::now();

        m_processLatencyMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        double totalFrameTime = std::chrono::duration<double>(endTime - m_lastFrameTime).count();
        m_lastFrameTime = endTime;
        if (totalFrameTime > 0.0) m_currentFPS = m_currentFPS * 0.85 + (1.0 / totalFrameTime) * 0.15;

        if (m_enableGUI) {
            if (m_isDragging && m_userDefinedRoi.width > 0 && m_userDefinedRoi.height > 0) {
                cv::rectangle(frame, m_userDefinedRoi, cv::Scalar(0, 255, 255), 2);
            }

            std::string origText = "ORIGINAL: " + std::to_string(static_cast<int>(m_originalFPS)) + " FPS";
            std::string procText = "PROCESS: " + std::to_string(static_cast<int>(m_currentFPS)) + " FPS (" + std::to_string(static_cast<int>(m_processLatencyMs)) + " ms)";

            cv::rectangle(frame, cv::Point(frame.cols - 520, 15), cv::Point(frame.cols - 15, 95), cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(frame, origText, cv::Point(frame.cols - 505, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::putText(frame, procText, cv::Point(frame.cols - 505, 83), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

            cv::resize(frame, m_displayFrame, cv::Size(1280, 720));
            cv::imshow(m_windowName, m_displayFrame);

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') break;
            else if (key == 'r' || key == 'R') {
                m_currentState = SystemState::DETECTION;
                m_trackerInitialized = false;
                m_detectionHistory.clear();
                m_hasUserRoi = false;
                m_gracePeriodCounter = 0;
            }
        }
    }

    m_capture.release();
    if (m_enableGUI) cv::destroyAllWindows();
}
void VideoPlayer::processFrame(cv::Mat& frame)
{
    if (m_currentState == SystemState::DETECTION) {
        m_currentDetections = m_detector.detect(frame);

        if (!m_currentDetections.empty()) {
            m_detectionHistory.push_back(m_currentDetections);
            if (m_detectionHistory.size() > m_maxHistoryFrames) m_detectionHistory.pop_front();
        }

        m_gracePeriodCounter++;
        if (m_gracePeriodCounter <= 3) {
            if (m_enableGUI) {
                for (const auto& det : m_currentDetections) {
                    cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
                }
                cv::putText(frame, "STATE: SCANNING (" + std::to_string(m_gracePeriodCounter) + "/3)", cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1.1, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
            }
            return;
        }

        if (m_hasTemplateImage && !m_currentDetections.empty()) {
            float bestTotalScore = -1.0f;
            Detection bestDet;
            bool foundMatch = false;

            cv::Mat templateEmbedding = ImageUtils::extractOsnetFeature(m_osnetNet, m_targetTemplateImage);
            float templateAspect = static_cast<float>(m_targetTemplateImage.cols) / m_targetTemplateImage.rows;

            for (const auto& det : m_currentDetections) {
                if (det.box.x < 0 || det.box.y < 0 || det.box.x + det.box.width > frame.cols || det.box.y + det.box.height > frame.rows) continue;

                cv::Mat roi = frame(det.box);
                if (roi.empty()) continue;

                float dummyScore = 0.0f;
                float totalScore = 0.0f;
                bool isDetVehicle = (det.classId != 0); // 0 = Person,others = Vehicle

                if (isDetVehicle) {
                    totalScore = VehicleMatcher::computeScore(m_targetTemplateImage, roi, static_cast<float>(det.box.width) / det.box.height, templateAspect, det.confidence, dummyScore);
                } else {
                    totalScore = PersonMatcher::computeScore(m_osnetNet, templateEmbedding, roi, static_cast<float>(det.box.width) / det.box.height, templateAspect, det.confidence, dummyScore);
                }

                if (totalScore > bestTotalScore) {
                    bestTotalScore = totalScore;
                    bestDet = det;
                    foundMatch = true;
                }
            }

            bool bestIsVehicle = (bestDet.classId != 0);
            float matchThreshold = bestIsVehicle ? 0.15f : 0.40f;

            if (foundMatch && bestTotalScore > matchThreshold) {
                m_selectedTarget = bestDet.box;
                m_targetClassId = bestDet.classId;
                m_targetClassName = (bestDet.classId == 0) ? "Person" : "Vehicle";
                m_targetAspectRatio = static_cast<float>(m_selectedTarget.width) / m_selectedTarget.height;
                m_targetInitialArea = static_cast<float>(m_selectedTarget.width * m_selectedTarget.height);
                m_lastKnownCenter = cv::Point2f(m_selectedTarget.x + m_selectedTarget.width / 2.0f, m_selectedTarget.y + m_selectedTarget.height / 2.0f);
                m_velocity = cv::Point2f(0.0f, 0.0f);
                m_occlusionFrameCounter = 0;
                m_gracePeriodCounter = 0;
                m_currentState = SystemState::TRACKING;
                m_trackerInitialized = false;
                return;
            }
        }

        if (m_enableGUI) {
            for (const auto& det : m_currentDetections) cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "STATE: DETECTION", cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1.1, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        }
    } else if (m_currentState == SystemState::TRACKING) {
        if (!m_trackerInitialized) {
            m_tracker.init(frame, m_selectedTarget);
            m_trackerInitialized = true;
            if (m_enableGUI) {
                cv::rectangle(frame, m_selectedTarget, cv::Scalar(0, 0, 255), 4);
                cv::putText(frame, "LOCKED: " + m_targetClassName, cv::Point(m_selectedTarget.x, std::max(m_selectedTarget.y - 12, 30)), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
            }
            return;
        }

        TrackResult result = m_tracker.update(frame);
        if (result.confidence < 0.45f) {
            m_trackerInitialized = false;
            m_hasUserRoi = false;
            m_currentState = SystemState::RE_DETECTION;
        } else {
            cv::Point2f currentCenter(result.box.x + result.box.width / 2.0f, result.box.y + result.box.height / 2.0f);
            cv::Point2f instantaneousVel = currentCenter - m_lastKnownCenter;
            if (cv::norm(instantaneousVel) < 1.0f) instantaneousVel = cv::Point2f(0, 0);

            m_velocity = m_velocity * 0.75f + instantaneousVel * 0.25f;
            m_selectedTarget = result.box;
            m_lastKnownCenter = currentCenter;

            if (m_enableGUI) {
                cv::rectangle(frame, m_selectedTarget, cv::Scalar(0, 0, 255), 4);
                cv::putText(frame, "LOCKED: " + m_targetClassName, cv::Point(m_selectedTarget.x, std::max(m_selectedTarget.y - 12, 30)), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
            }
        }

        if (m_enableGUI) {
            cv::putText(frame, "STATE: TRACKING", cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(0, 255, 255), 3, cv::LINE_AA);
        }
    } else if (m_currentState == SystemState::RE_DETECTION) {
        if (!m_hasUserRoi) {
            if (m_enableGUI) {
                cv::putText(frame, "STATE: RE_DETECTION (DRAW ROI)", cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
            }
            return;
        }

        auto detections = m_detector.detect(frame);
        cv::Rect bestMatchBox;
        float bestTotalScore = -1.0f;
        bool found = false;
        bool targetIsVehicle = (m_targetClassId != 0);

        cv::Mat templateEmbedding = targetIsVehicle ? cv::Mat() : ImageUtils::extractOsnetFeature(m_osnetNet, m_targetTemplateImage);
        float templateAspect = static_cast<float>(m_targetTemplateImage.cols) / m_targetTemplateImage.rows;

        for (const auto& det : detections) {
            cv::Point2f detCenter(det.box.x + det.box.width / 2.0f, det.box.y + det.box.height / 2.0f);
            if (!m_userDefinedRoi.contains(detCenter)) continue;

            bool isDetVehicle = (det.classId != 0);
            if (targetIsVehicle != isDetVehicle) continue;

            cv::Mat roi = frame(det.box);
            if (roi.empty()) continue;

            float dummyScore = 0.0f;
            float totalScore = 0.0f;

            if (targetIsVehicle) {
                totalScore = VehicleMatcher::computeScore(m_targetTemplateImage, roi, static_cast<float>(det.box.width) / det.box.height, templateAspect, det.confidence, dummyScore);
            } else {
                totalScore = PersonMatcher::computeScore(m_osnetNet, templateEmbedding, roi, static_cast<float>(det.box.width) / det.box.height, templateAspect, det.confidence, dummyScore);
            }

            if (totalScore > bestTotalScore) {
                bestTotalScore = totalScore;
                bestMatchBox = det.box;
                found = true;
            }
        }

        float reDetectionThreshold = targetIsVehicle ? 0.18f : 0.40f;
        if (found && bestTotalScore > reDetectionThreshold) {
            m_selectedTarget = bestMatchBox;
            m_lastKnownCenter = cv::Point2f(m_selectedTarget.x + m_selectedTarget.width / 2.0f, m_selectedTarget.y + m_selectedTarget.height / 2.0f);
            m_tracker.init(frame, m_selectedTarget);
            m_trackerInitialized = true;
            m_currentState = SystemState::TRACKING;
            m_hasUserRoi = false;
        } else if (m_enableGUI) {
            cv::rectangle(frame, m_userDefinedRoi, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, "NO MATCH IN ROI", cv::Point(30, 100), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
    }
}

