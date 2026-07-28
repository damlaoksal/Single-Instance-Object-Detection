#include "PersonMatcher.h"
#include "ImageUtils.h"
#include <algorithm>
#include <cmath>

float PersonMatcher::computeScore(cv::dnn::Net& osnetNet, const cv::Mat& templateEmbedding, const cv::Mat& roi, float candAspect, float templateAspect, float conf, float& outReidScore)
{
    cv::Mat candidateEmbedding = ImageUtils::extractOsnetFeature(osnetNet, roi);
    float reidScore = ImageUtils::calculateCosineSimilarity(templateEmbedding, candidateEmbedding);
    reidScore = std::max(0.0f, reidScore);
    outReidScore = reidScore;

    float aspectDiff = std::abs(candAspect - templateAspect) / (templateAspect + 1e-5f);
    float aspectScore = std::max(0.0f, 1.0f - (aspectDiff * 0.5f));

    return (0.70f * reidScore) + (0.15f * conf) + (0.15f * aspectScore);
}