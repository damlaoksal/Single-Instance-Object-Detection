#pragma once

#include <opencv2/opencv.hpp>
#include "Detection.h"

/**
 * @brief LAB Color Space and Histogram-based matching module for vehicles.
 */
class VehicleMatcher {
public:
    static float computeScore(const cv::Mat& templateImg, const cv::Mat& roi, float candAspect, float templateAspect, float conf, float& outColorScore);
};