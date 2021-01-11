#include <stdio.h>
#include <iostream>
#include "lib/CameraStabilization/CameraStabilization.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"

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

    cv::Size originalSize{1920, 1200};
    std::vector<double> scaleFactors;
    std::vector<providentia::calibration::dynamic::SurfBFDynamicCalibrator> calibrators;
    std::vector<providentia::opticalflow::DenseOpticalFlow> opticalFlows;
    opticalFlows.emplace_back();
    for (int i = 5; i <= 10; i += 1) {
//    {
//        int i = 10;
        double scaleFactor = i / 10.0;
        scaleFactors.emplace_back(scaleFactor);
        calibrators.emplace_back(1000, cv::NORM_L2, true);
        calibrators[calibrators.size() - 1].setScaleFactor(
                cv::Size(originalSize.width * scaleFactor, originalSize.height * scaleFactor));
        opticalFlows.emplace_back();
    }

    int padding = 20;

    std::string windowName = "Dynamic Camera Stabilization";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    double calculationScaleFactor = 1;
    double renderingScaleFactor = 0.5;
    renderingScaleFactor /= calculationScaleFactor;

    bool writeOpticalFlow = true;
    std::string middleSuffix = "testing";
    providentia::utils::CsvWriter magnitudeCsv(basePath + filename + "_opticalflow_" + middleSuffix + ".csv",
                                               writeOpticalFlow);
    providentia::utils::CsvWriter durationCsv(basePath + filename + "_durations_" + middleSuffix + ".csv",
                                              writeOpticalFlow);

    magnitudeCsv.append("Timestamp", "Milliseconds", "<Original> Opt. Flow [px]");
    durationCsv.append("Timestamp", "Milliseconds", "<Original> Frame Time [ms]");
    for (const auto &scaleFactor : scaleFactors) {
        magnitudeCsv.append("<" + std::to_string(scaleFactor) + "> Opt. Flow [px]");
        durationCsv.append("<" + std::to_string(scaleFactor) + "> Frame Time [ms]");
    }
    magnitudeCsv.newLine();
    durationCsv.newLine();

    auto start = providentia::utils::TimeMeasurable::now().count();
    int frameNumber = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        cv::resize(frame, frame, cv::Size(), calculationScaleFactor, calculationScaleFactor);
        cv::Mat originalFrame = frame.clone();
        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(frame);

        auto now = providentia::utils::TimeMeasurable::now().count();
        long duration = 0;
        std::vector<cv::Mat> colorFrames{providentia::utils::pad(originalFrame, padding)};
        std::vector<cv::Mat> backgroundMasks{
                providentia::utils::pad(cv::Mat::ones(originalFrame.size(), CV_8UC1) * 255, padding)};
        durationCsv.append(now, now - start);
        durationCsv.append(0);
        for (int i = 0; i < calibrators.size(); i++) {
            calibrators[i].stabilize(gpu_frame);
            cv::Mat currentFrame = cv::Mat(calibrators[i].getStabilizedFrame());
            cv::resize(currentFrame, currentFrame, originalSize);
            currentFrame = providentia::utils::pad(currentFrame, padding);
            providentia::utils::addText(currentFrame, std::to_string(scaleFactors[i]), 10, 30);
            long currentDuration = calibrators[i].getTotalMilliseconds();
            duration += currentDuration;
            durationCsv.append(currentDuration);
            providentia::utils::addText(currentFrame, std::to_string(currentDuration) + " ms", 10, 60);
            colorFrames.emplace_back(currentFrame);
            backgroundMasks.emplace_back(providentia::utils::pad(cv::Mat(calibrators[i].getLatestMask()), padding));
        }
        durationCsv.newLine();

        magnitudeCsv.append(now, now - start);
        std::vector<cv::Mat> opticalFlowFrames;
        for (int i = 0; i < opticalFlows.size(); i++) {
            cv::Mat currentFrame = opticalFlows[i].calculate(colorFrames[i]);
            if (i > 0) {
                providentia::utils::addText(currentFrame, std::to_string(scaleFactors[i - 1]), 10, 30);
            }
            double magnitude = opticalFlows[i].getMagnitudeMean();
            providentia::utils::addText(currentFrame,
                                        std::to_string(magnitude) + " px", 10,
                                        60);
            long currentDuration = opticalFlows[i].getTotalMilliseconds();
            duration += currentDuration;
            providentia::utils::addText(currentFrame, std::to_string(currentDuration) + "  ms", 10, 90);
            opticalFlowFrames.emplace_back(currentFrame);

            magnitudeCsv.append(magnitude);
        }
        magnitudeCsv.newLine();

        for (auto &frame : colorFrames) {
            cv::resize(frame, frame, cv::Size(), renderingScaleFactor, renderingScaleFactor);
        }

        for (auto &frame : opticalFlowFrames) {
            cv::resize(frame, frame, colorFrames[0].size());
        }
        for (auto &frame : backgroundMasks) {
            cv::resize(frame, frame, colorFrames[0].size());
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }

        cv::Mat colorFramesMerged;
        cv::hconcat(colorFrames, colorFramesMerged);
        cv::Mat backgroundMasksFramesMerged;
        cv::hconcat(backgroundMasks, backgroundMasksFramesMerged);
        cv::Mat opticalFlowFramesMerged;
        cv::hconcat(opticalFlowFrames, opticalFlowFramesMerged);

        cv::Mat finalFrame;
        cv::vconcat(std::vector<cv::Mat>{colorFramesMerged, backgroundMasksFramesMerged, opticalFlowFramesMerged},
                    finalFrame);

//        cv::resize(finalFrame, finalFrame, cv::Size(), renderingScaleFactor, renderingScaleFactor);
        cv::imshow(windowName, finalFrame);

        std::cout << frameNumber++ << std::endl;

        if ((char) cv::waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
