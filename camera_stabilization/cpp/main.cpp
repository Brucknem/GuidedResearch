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

#include "DynamicCalibration.hpp"

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
    providentia::calibration::dynamic::SurfBFDynamicCalibrator calibrator(1000, cv::NORM_L2, 1);
    int padding = 0;

    std::string windowName = "Dynamic Camera Stabilization";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
//
//    std::string matchingWindowName = "Matching Camera Stabilization";
//    cv::namedWindow(matchingWindowName, cv::WINDOW_AUTOSIZE);


    double calculationScaleFactor = 1;
    double renderingScaleFactor = 0.65;

    std::stringstream frameText;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        frameText.str(std::string());
        cv::Mat originalFrame = frame.clone();
        cv::resize(frame, frame, cv::Size(), calculationScaleFactor, calculationScaleFactor);

        if (!calibrator.hasReferenceFrame()) {
            calibrator.setReferenceFrame(frame);
        }

        cv::Mat stabilized = calibrator.stabilize(frame);

        stabilized = cv::Mat(stabilized,
                             cv::Rect(padding, padding, stabilized.cols - 2 * padding, stabilized.rows - 2 * padding));
        originalFrame = cv::Mat(originalFrame, cv::Rect(padding, padding, originalFrame.cols - 2 * padding,
                                                        originalFrame.rows - 2 * padding));
        cv::hconcat(std::vector<cv::Mat>{originalFrame, stabilized}, stabilized);
        cv::resize(stabilized, stabilized, cv::Size(), renderingScaleFactor, renderingScaleFactor);

//        cv::Mat flannMatching = flannResult[0];
//        cv::resize(flannMatching, flannMatching, cv::Size(), renderingScaleFactor, renderingScaleFactor);

        long duration = 0;
        duration += calibrator.getRuntime();
        double fps = 1000.0 / duration;

        frameText << "Duration: " << duration << "ms - FPS: " << fps;
        cv::putText(stabilized, frameText.str(), cv::Point(10, 30),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 250), 1, 8);
        cv::imshow(windowName, stabilized);
        // cv::imshow(matchingWindowName, flannMatching);

        if ((char) cv::waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
