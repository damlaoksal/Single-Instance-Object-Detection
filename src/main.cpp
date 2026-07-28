#include "VideoPlayer.h"
#include <filesystem>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    std::cout << "Current directory: " << std::filesystem::current_path() << std::endl;

    VideoPlayer player("../../videos/car2.mp4", true);
    player.setTargetTemplate("../../target_input_images/car2.png");

    if (!player.initialize())
        return -1;

    player.run();

    return 0;
}