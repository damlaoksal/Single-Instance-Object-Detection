#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

/**
 * @brief Image processing, OSNet embedding, and cosine similarity helper utilities.
 */
namespace ImageUtils {
    cv::Mat extractOsnetFeature(cv::dnn::Net& net, const cv::Mat& inputImg);
    float calculateCosineSimilarity(const cv::Mat& v1, const cv::Mat& v2);
}