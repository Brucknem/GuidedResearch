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
#include "BackgroundSegmentation.h"
#include "Utils.hpp"

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
    providentia::calibration::dynamic::SurfBFDynamicCalibrator calibrator_downsampled(1000, cv::NORM_L2);

    int opticalFlowVerbosity = 0;
    providentia::opticalflow::DenseOpticalFlow opticalFlow_original(0);
    providentia::opticalflow::DenseOpticalFlow opticalFlow_stabilized(0);
    providentia::opticalflow::DenseOpticalFlow opticalFlow_stabilized_extended(0);

    int padding = 10;

    std::string windowName = "Dynamic Camera Stabilization";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    double calculationScaleFactor = 1;
    double renderingScaleFactor = 0.5;
    renderingScaleFactor /= calculationScaleFactor;

    bool writeOpticalFlow = true;
    std::ofstream magnitudeCsv;
    if (writeOpticalFlow) {
        std::string magnitudeCsvName = basePath + filename + "_opticalflow.csv";
        magnitudeCsv.open(magnitudeCsvName);
        magnitudeCsv << "Timestamp,Milliseconds,Original [px],Stabilized [px],Stabilized (Extended) [px]" << std::endl;
        magnitudeCsv.close();
        magnitudeCsv.open(magnitudeCsvName, std::ios_base::app); // append instead of overwrite
    }

    auto start = providentia::utils::TimeMeasurable::now().count();
    cv::Size originalSize;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        cv::resize(frame, frame, cv::Size(), calculationScaleFactor, calculationScaleFactor);
        cv::Mat originalFrame = frame.clone();
        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(frame);

        originalSize = frame.size();
        calibrator_downsampled.setScaleFactor(cv::Size(originalSize.width * 0.5, originalSize.height * 0.5));

        calibrator.stabilize(gpu_frame);
//        calibrator_extended.stabilize(gpu_frame);
        calibrator_downsampled.stabilize(gpu_frame);
        cv::Mat stabilized = cv::Mat(calibrator.getStabilizedFrame());
//        cv::Mat stabilized_extended = cv::Mat(calibrator_extended.getStabilizedFrame());
        cv::Mat stabilized_extended = cv::Mat(calibrator_downsampled.getStabilizedFrame());
        cv::resize(stabilized_extended, stabilized_extended, originalSize);

        stabilized = providentia::utils::pad(stabilized, padding);
        stabilized_extended = providentia::utils::pad(stabilized_extended, padding);
//        std::cout << calibrator.durations_str() << std::endl;
        originalFrame = providentia::utils::pad(cv::Mat(originalFrame), padding);

        cv::Mat finalFrame;

        cv::Mat colorFrames;
        cv::hconcat(std::vector<cv::Mat>{originalFrame, stabilized, stabilized_extended}, colorFrames);

//        cv::Mat calibratorFrames;
//        cv::Mat referenceFrame = cv::Mat(calibrator.getReferenceFrame());
//        cv::cvtColor(referenceFrame, referenceFrame, cv::COLOR_GRAY2BGR);
//        cv::hconcat(std::vector<cv::Mat>{stabilized, pad(referenceFrame, padding)},
//                    calibratorFrames);
//        cv::Mat calibratorMasks;
//        cv::hconcat(std::vector<cv::Mat>{pad(cv::Mat(calibrator.getLatestMask()), padding),
//                                         pad(cv::Mat(calibrator.getReferenceMask()), padding)},
//                    calibratorMasks);
//        cv::cvtColor(calibratorMasks, calibratorMasks, cv::COLOR_GRAY2BGR);
//
        stabilized = opticalFlow_stabilized.calculate(stabilized);
        stabilized_extended = opticalFlow_stabilized_extended.calculate(stabilized_extended);
        originalFrame = opticalFlow_original.calculate(originalFrame);

        cv::Mat opticalFlowFrames;
        cv::hconcat(std::vector<cv::Mat>{originalFrame, stabilized, stabilized_extended}, opticalFlowFrames);
        cv::resize(opticalFlowFrames, opticalFlowFrames, colorFrames.size());

        if (writeOpticalFlow) {
            auto now = providentia::utils::TimeMeasurable::now().count();
            magnitudeCsv << now << "," << now - start << "," << opticalFlow_original.getMagnitudeMean() << ","
                         << opticalFlow_stabilized.getMagnitudeMean() << ","
                         << opticalFlow_stabilized_extended.getMagnitudeMean() << std::endl;
        }
//        cv::vconcat(std::vector<cv::Mat>{calibratorFrames, calibratorMasks}, finalFrame);
//        cv::vconcat(std::vector<cv::Mat>{colorFrames,}, finalFrame);
        cv::vconcat(std::vector<cv::Mat>{colorFrames, opticalFlowFrames}, finalFrame);

//        calibrator.draw();

        cv::resize(finalFrame, finalFrame, cv::Size(), renderingScaleFactor, renderingScaleFactor);

        providentia::utils::addText(finalFrame,
                                    providentia::utils::durationInfo("Calibration", calibrator.getTotalMilliseconds()),
                                    10, 30);
        providentia::utils::addText(finalFrame, providentia::utils::durationInfo("Calibration Downsampled",
                                                                                 calibrator_downsampled.getTotalMilliseconds()),
                                    10,
                                    60);
        providentia::utils::addText(finalFrame, providentia::utils::durationInfo("Total",
                                                                                 calibrator.getTotalMilliseconds()), 10,
                                    90);


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
