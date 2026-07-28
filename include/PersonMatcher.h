#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "Detection.h"

/**
 * @brief OSNet Re-ID based matching module for persons.
 */
class PersonMatcher {
public:
    static float computeScore(cv::dnn::Net& osnetNet, const cv::Mat& templateEmbedding, const cv::Mat& roi, float candAspect, float templateAspect, float conf, float& outReidScore);
};