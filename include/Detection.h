#pragma once

#include <opencv2/opencv.hpp>

/**
 * @brief Holds the essential information of a detected object.
 */
struct Detection {
    cv::Rect box;          // Bounding box coordinates in the image
    float confidence;      // Confidence score of the detection [0.0 - 1.0]
    int classId;           // Identifier for the detected object class
};