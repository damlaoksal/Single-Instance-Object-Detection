#include "ImageUtils.h"

namespace ImageUtils {

cv::Mat extractOsnetFeature(cv::dnn::Net& net, const cv::Mat& inputImg)
{
    if (inputImg.empty()) return cv::Mat();

    cv::Mat blob = cv::dnn::blobFromImage(
        inputImg, 
        1.0 / 255.0, 
        cv::Size(128, 256), 
        cv::Scalar(0.485 * 255, 0.456 * 255, 0.406 * 255), 
        true, 
        false
    );
    
    net.setInput(blob);
    cv::Mat embedding = net.forward();
    return embedding.reshape(1, 1).clone();
}

float calculateCosineSimilarity(const cv::Mat& v1, const cv::Mat& v2)
{
    if (v1.empty() || v2.empty()) return 0.0f;
    double dotProduct = v1.dot(v2);
    double norm1 = cv::norm(v1);
    double norm2 = cv::norm(v2);
    if (norm1 == 0.0 || norm2 == 0.0) return 0.0f;
    return static_cast<float>(dotProduct / (norm1 * norm2));
}

} // namespace ImageUtils