//
// Created by brucknem on 21.12.20.
//

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include <opencv2/cudafeatures2d.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/cudaoptflow.hpp"
#include <chrono>
#include <thread>

#include "Utils.hpp"

#ifndef DYNAMICSTABILIZATION_OPTICALFLOW_H
#define DYNAMICSTABILIZATION_OPTICALFLOW_H

namespace providentia {
    namespace opticalflow {
        class DenseOpticalFlow : public providentia::utils::TimeMeasurable {
        private:
            cv::cuda::GpuMat currentFrame, previousFrame, denseOpticalFlow;
            cv::Mat hsv, denseOpticalFlow_cpu, magnitude, angle;
            std::vector<cv::Mat> flowParts{3};
            cv::Mat magnitudes, angles;
            cv::Ptr<cv::cuda::FarnebackOpticalFlow> farnbackOpticalFlow;
            cv::cuda::Stream stream;
            cv::Mat _hsv[3], hsv8, bgr;

            void initialize(const cv::Mat &frame) {
                previousFrame.upload(frame);
                cv::cuda::cvtColor(previousFrame, previousFrame, cv::COLOR_BGR2GRAY);
                hsv = cv::Mat(frame.size(), CV_8UC3, cv::Scalar(255));
            }

        public:
            explicit DenseOpticalFlow(int verbosity = 0) : providentia::utils::TimeMeasurable("Dense Optical Flow",
                                                                                              verbosity) {
                farnbackOpticalFlow = cv::cuda::FarnebackOpticalFlow::create();
            }

            cv::Mat calculate(const cv::Mat &frame) {
                if (previousFrame.empty()) {
                    initialize(frame);
                    return cv::Mat(frame.size(), CV_8UC3, cv::Scalar(0));
                }

                clear();
                currentFrame.upload(frame);
                cv::cuda::cvtColor(currentFrame, currentFrame, cv::COLOR_BGR2GRAY);
                addTimestamp("Frame to grayscale", 2);

                farnbackOpticalFlow->calc(previousFrame, currentFrame, denseOpticalFlow, stream);
                addTimestamp("Calculated optical flow", 2);
                denseOpticalFlow.download(denseOpticalFlow_cpu);
                addTimestamp("Downloaded frame", 2);

                cv::split(denseOpticalFlow_cpu, flowParts);
                addTimestamp("Split in channels", 2);

                cv::cartToPolar(flowParts[0], flowParts[1], flowParts[0], flowParts[1], true);
                addTimestamp("To polar coordinates", 2);
                magnitude = flowParts[0];
                angle = flowParts[1];
                normalize(flowParts[0], flowParts[2], 0.0f, 1.0f, cv::NORM_MINMAX);
                addTimestamp("Normalized magnitude", 2);
                flowParts[1] *= ((1.f / 360.f) * (180.f / 255.f));
                addTimestamp("Calculated angles", 2);
                //build hsv image
                _hsv[0] = flowParts[1];
                _hsv[1] = cv::Mat::ones(flowParts[1].size(), CV_32F);
                _hsv[2] = flowParts[2];
                merge(_hsv, 3, hsv);
                addTimestamp("Merged channels", 2);

                hsv.convertTo(hsv8, CV_8U, 255.0);
                cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
                addTimestamp("Conversion to BGR", 1);

                previousFrame = currentFrame;
                return bgr;
            }

            double getMagnitudeMean() {
                return cv::mean(magnitude)[0];
            }

            double getAngleMean() {
                return cv::mean(angle)[0];
            }
        };
    }
}

#endif //DYNAMICSTABILIZATION_OPTICALFLOW_H
