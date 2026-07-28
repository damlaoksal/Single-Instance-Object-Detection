#include "VehicleMatcher.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

float VehicleMatcher::computeScore(const cv::Mat& templateImg, const cv::Mat& roi, float candAspect, float templateAspect, float conf, float& outColorScore)
{
    // Cache template histogram to prevent redundant CPU computations on Edge devices
    static cv::Mat cachedTemplateImg;
    static cv::Mat cachedTemplateHist;

    int channels[] = {0, 1, 2};
    int histSize[] = {24, 24, 24};
    float lRange[] = {0, 256};
    float aRange[] = {0, 256};
    float bRange[] = {0, 256};
    const float* ranges[] = {lRange, aRange, bRange};

    // Only compute the template histogram if it's the first time or if the template changes
    if (cachedTemplateImg.empty() || cachedTemplateImg.data != templateImg.data) {
        cv::Mat targetForHist = templateImg;
        if (templateImg.cols > 64 || templateImg.rows > 64) {
            cv::resize(templateImg, targetForHist, cv::Size(64, 64), 0, 0, cv::INTER_AREA);
        }
        cv::Mat templateLab;
        cv::cvtColor(targetForHist, templateLab, cv::COLOR_BGR2Lab);
        cv::calcHist(&templateLab, 1, channels, cv::Mat(), cachedTemplateHist, 3, histSize, ranges, true, false);
        cv::normalize(cachedTemplateHist, cachedTemplateHist, 1.0, 0.0, cv::NORM_L1, -1, cv::Mat());
        
        cachedTemplateImg = templateImg;
    }

    cv::Mat roiForHist = roi;
    if (roi.cols > 64 || roi.rows > 64) {
        cv::resize(roi, roiForHist, cv::Size(64, 64), 0, 0, cv::INTER_AREA);
    }

    cv::Mat roiLab, roiHist;
    cv::cvtColor(roiForHist, roiLab, cv::COLOR_BGR2Lab);
    cv::calcHist(&roiLab, 1, channels, cv::Mat(), roiHist, 3, histSize, ranges, true, false);
    cv::normalize(roiHist, roiHist, 1.0, 0.0, cv::NORM_L1, -1, cv::Mat());

    float colorScore = static_cast<float>(cv::compareHist(cachedTemplateHist, roiHist, cv::HISTCMP_INTERSECT));
    colorScore = std::min(1.0f, std::max(0.0f, colorScore));
    outColorScore = colorScore;

    float aspectDiff = std::abs(candAspect - templateAspect) / (templateAspect + 1e-5f);
    float aspectScore = std::max(0.0f, 1.0f - (aspectDiff * 0.5f));

    return (0.85f * colorScore) + (0.10f * conf) + (0.05f * aspectScore);
}