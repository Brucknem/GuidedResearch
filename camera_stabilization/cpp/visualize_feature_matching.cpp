#include <stdio.h>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include <opencv2/cudafeatures2d.hpp>
#include "opencv2/calib3d.hpp"
#include <chrono>
#include <thread>
#include <fstream>

#include "DynamicCalibration.hpp"
#include "OpticalFlow.h"
#include "CSVWriter.hpp"


int main(int argc, char const *argv[]) {
    cv::cuda::Stream cudaStream;
    std::string basePath = "/mnt/local_data/providentia/test_recordings/videos/";
    // cv::VideoCapture cap("/mnt/local_data/providentia/test_recordings/videos/s40_n_far_image_raw.mp4");
    std::string filename = "s40_n_far_image_raw";
    std::string suffix = ".mp4";
    cv::VideoCapture cap(basePath + filename + suffix);
    if (!cap.isOpened()) // if not success, exit program
    {
        std::cout << "Cannot open the video." << std::endl;
        return -1;
    }

    cv::Mat frame;
//    providentia::calibration::dynamic::ExtendedSurfBFDynamicCalibrator calibrator_extended(1000, cv::NORM_L2);
    providentia::calibration::dynamic::SurfBFDynamicCalibrator calibrator(1000, cv::NORM_L2);

    int padding = 10;

    std::string windowName = "Dynamic Camera Stabilization";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    double calculationScaleFactor = 1;
    double renderingScaleFactor = 0.5;
    renderingScaleFactor /= calculationScaleFactor;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        cv::resize(frame, frame, cv::Size(), calculationScaleFactor, calculationScaleFactor);
        cv::Mat originalFrame = frame.clone();
        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(frame);

//        calibrator.stabilize(gpu_frame, cv::INTER_LINEAR);
        calibrator.stabilize(gpu_frame, cv::INTER_CUBIC);
//        calibrator.stabilize(gpu_frame, cv::INTER_NEAREST);
        cv::Mat stabilized = cv::Mat(calibrator.getStabilizedFrame());

        cv::resize(stabilized, stabilized, cv::Size(), renderingScaleFactor, renderingScaleFactor);
        cv::resize(originalFrame, originalFrame, cv::Size(), renderingScaleFactor, renderingScaleFactor);

        stabilized = providentia::utils::pad(stabilized, padding);
        originalFrame = providentia::utils::pad(originalFrame, padding);

        cv::Mat finalFrame;
        cv::hconcat(std::vector<cv::Mat>{originalFrame, stabilized}, finalFrame);

        providentia::utils::addText(finalFrame,
                                    providentia::utils::durationInfo("Calibration", calibrator.getTotalMilliseconds()),
                                    10, 30);
        providentia::utils::addText(finalFrame, providentia::utils::durationInfo("Total",
                                                                                 calibrator.getTotalMilliseconds()), 10,
                                    60);


        cv::imshow(windowName, finalFrame);
        calibrator.draw(renderingScaleFactor);

        if ((char) cv::waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
