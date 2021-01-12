#include "ImageUtils.hpp"
#include <opencv2/opencv.hpp>

ImageBasedVideoCapture::ImageBasedVideoCapture(std::string path, std::string fileEnding, int frameRate, int maxLoadedFrames, bool loop) : path(path), frameRate(frameRate), loop(loop)
{
    this->frameTime = 1.0 / frameRate;
};

cv::Mat ImageBasedVideoCapture::read(){
    return cv::Mat();
}

int ImageBasedVideoCapture::getNumberOfFrames(){
    return 0;
}