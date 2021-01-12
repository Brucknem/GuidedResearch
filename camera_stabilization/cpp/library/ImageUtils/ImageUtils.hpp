#ifndef __ImageUtils_hpp_
#define __ImageUtils_hpp_

#include <opencv2/opencv.hpp>
#include <map>
#include <vector>
#include <thread>

class VideoCapture
{
public:
    virtual cv::Mat read() = 0;
    virtual int getNumberOfFrames() = 0;
};

class ImageBasedVideoCapture : public VideoCapture
{
private:
    std::string path;
    std::vector<std::string> fileNames;
    std::map<std::string, cv::Mat> frames; 
    int frameIndex = 0;
    int frameRate;
    float frameTime;
    bool loop;
    int latestLoadedFrame = 0;
    bool isThreadRunning = false;
    std::thread loadingThread;

public:
    ImageBasedVideoCapture(std::string path, std::string fileEnding = ".png", int frameRate = 25, int maxLoadedFrames = 100, bool loop = true);
    cv::Mat read() override;
    int getNumberOfFrames() override;
};

#endif // __ImageUtils_hpp_