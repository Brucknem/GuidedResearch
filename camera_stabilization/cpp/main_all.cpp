#include <stdio.h>
#include <iostream>
#include "lib/CameraStabilization/CameraStabilization.hpp"
#include "lib/ImageUtils/ImageUtils.hpp"
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

std::string durationInfo(const std::string &name, long milliseconds) {
    std::stringstream ss;
    ss << name << "- Duration: " << milliseconds << "ms - FPS: " << 1000. / milliseconds;
    return ss.str();
}

void addText(cv::Mat &frame, std::string text, int x, int y) {
    cv::putText(frame, text, cv::Point(x, y),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 0), 2, cv::FONT_HERSHEY_SIMPLEX);
}

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

    std::string magnitudeCsvName = basePath + filename + "_opticalflow.csv";
    std::ofstream magnitudeCsv;
    magnitudeCsv.open(magnitudeCsvName);
    magnitudeCsv << "Timestamp,Milliseconds,Original [px],Stabilized [px]" << std::endl;
    magnitudeCsv.close();
    magnitudeCsv.open(magnitudeCsvName, std::ios_base::app); // append instead of overwrite

    cv::Mat frame;
    providentia::calibration::dynamic::SurfBFDynamicCalibrator calibrator(1000, cv::NORM_L2, 1);
    providentia::opticalflow::DenseOpticalFlow opticalFlow_original(2);
    providentia::opticalflow::DenseOpticalFlow opticalFlow_stabilized(2);

    int padding = 10;

    std::string windowName = "Dynamic Camera Stabilization";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
//
//    std::string matchingWindowName = "Matching Camera Stabilization";
//    cv::namedWindow(matchingWindowName, cv::WINDOW_AUTOSIZE);


    double calculationScaleFactor = 1;
    double renderingScaleFactor = 0.65;


    cv::Ptr<cv::BackgroundSubtractor> backgroundSubtractor_original = cv::createBackgroundSubtractorMOG2();
    cv::Ptr<cv::BackgroundSubtractor> backgroundSubtractor_stabilized = cv::createBackgroundSubtractorMOG2();
    cv::Mat foregroundMask_original, foregroundMask_stabilized;

    auto start = providentia::utils::TimeMeasurable::now().count();

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        cv::resize(frame, frame, cv::Size(), calculationScaleFactor, calculationScaleFactor);
        cv::Mat originalFrame = frame.clone();
        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(frame);

        v::Mat stabilized = ccalibrator.stabilize(gpu_frame);
        stabilized = cv::Mat(stabilized,
                             cv::Rect(padding, padding, stabilized.cols - 2 * padding, stabilized.rows - 2 * padding));
        originalFrame = cv::Mat(originalFrame, cv::Rect(padding, padding, originalFrame.cols - 2 * padding,
                                                        originalFrame.rows - 2 * padding));

        backgroundSubtractor_stabilized->apply(stabilized, foregroundMask_stabilized);
        backgroundSubtractor_original->apply(originalFrame, foregroundMask_original);

        cv::Mat colorFrames;
        cv::hconcat(std::vector<cv::Mat>{originalFrame, stabilized}, colorFrames);
        stabilized = opticalFlow_stabilized.calculate(stabilized);
        originalFrame = opticalFlow_original.calculate(originalFrame);

        auto now = providentia::utils::TimeMeasurable::now().count();
//        magnitudeCsv << now << "," << now - start << "," << opticalFlow_original.getMagnitudeMean() << ","
//                     << opticalFlow_stabilized.getMagnitudeMean() << std::endl;

        cv::Mat opticalFlowFrames;
        cv::hconcat(std::vector<cv::Mat>{originalFrame, stabilized}, opticalFlowFrames);

        cv::Mat backgroundFrames;
        cv::hconcat(std::vector<cv::Mat>{foregroundMask_original, foregroundMask_stabilized}, backgroundFrames);
        cv::cvtColor(backgroundFrames, backgroundFrames, cv::COLOR_GRAY2BGR);

        cv::Mat finalFrame;
        cv::vconcat(std::vector<cv::Mat>{colorFrames, opticalFlowFrames, backgroundFrames}, finalFrame);

        finalFrame = backgroundFrames;
        cv::resize(finalFrame, finalFrame, cv::Size(), renderingScaleFactor, renderingScaleFactor);

        addText(finalFrame, durationInfo("Calibration", calibrator.getTotalMilliseconds()), 10, 30);
        addText(finalFrame, durationInfo("Optical Flow (original)", opticalFlow_original.getTotalMilliseconds()), 10,
                60);
        addText(finalFrame, durationInfo("Optical Flow (stabilized)", opticalFlow_stabilized.getTotalMilliseconds()),
                10, 90);
        addText(finalFrame, durationInfo("Total",
                                         calibrator.getTotalMilliseconds() +
                                         opticalFlow_original.getTotalMilliseconds() +
                                         opticalFlow_stabilized.getTotalMilliseconds()), 10, 120);

        cv::imshow(windowName, finalFrame);
        // cv::imshow(matchingWindowName, flannMatching);

        if ((char) cv::waitKey(1) == 27) {
            break;
        }
    }

    magnitudeCsv.close();
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
