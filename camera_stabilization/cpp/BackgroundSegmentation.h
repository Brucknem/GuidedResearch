//
// Created by brucknem on 22.12.20.
//
#include <stdio.h>
#include <iostream>
#include "lib/CameraStabilization/CameraStabilization.hpp"
#include "lib/ImageUtils/ImageUtils.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <utility>
#include <opencv2/video.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include <opencv2/cudafeatures2d.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/bgsegm.hpp"
#include <chrono>
#include <thread>
#include <fstream>

#ifndef DYNAMICSTABILIZATION_BACKGROUNDSEGMENTATION_H
#define DYNAMICSTABILIZATION_BACKGROUNDSEGMENTATION_H

namespace providentia {
    namespace segmentation {

        class BackgroundSegmentation {
        protected:
            cv::Ptr<cv::BackgroundSubtractor> algorithm;
            cv::Mat foregroundMask;
            cv::Mat backgroundMask;
            bool backgroundNeedsReset = true;

        public:
            explicit BackgroundSegmentation(cv::Ptr<cv::BackgroundSubtractor> _algorithm) : algorithm(std::move(
                    _algorithm)) {}

            virtual void apply(const cv::Mat &frame) {
                algorithm->apply(frame, foregroundMask);
                setForegroundUpdated();
                postProcess();
            };

            virtual cv::Mat getForegroundMask() {
                return foregroundMask;
            }

            virtual cv::Mat getBackgroundMask() {
                if (backgroundNeedsReset) {
                    backgroundMask = 255 - foregroundMask;
                    backgroundNeedsReset = false;
                }
                return backgroundMask;
            }

            void setForegroundUpdated() {
                backgroundNeedsReset = true;
            }

            virtual void postProcess() = 0;
        };

        class CudaBackgroundSegmentation : public BackgroundSegmentation {
        protected:
            explicit CudaBackgroundSegmentation(const cv::Ptr<cv::BackgroundSubtractorMOG2> &_algorithm)
                    : BackgroundSegmentation(
                    _algorithm) {}

            cv::cuda::Stream stream;
            cv::cuda::GpuMat foregroundMask_gpu;
            bool foregroundNeedsReset = true;

            std::vector<cv::Ptr<cv::cuda::Filter>> filters;

        public:
            void apply(const cv::cuda::GpuMat &frame) {
                applyGpu(frame);
                postProcess();
                foregroundNeedsReset = true;
                setForegroundUpdated();
            };

            virtual void applyGpu(const cv::cuda::GpuMat &frame) = 0;

            cv::Mat getForegroundMask() override {
                if (foregroundNeedsReset) {
                    stream.waitForCompletion();
                    foregroundMask_gpu.download(foregroundMask);
                    foregroundNeedsReset = false;
                    backgroundNeedsReset = true;
                }
                return BackgroundSegmentation::getForegroundMask();
            }

            cv::Mat getBackgroundMask() override {
                if (foregroundNeedsReset) {
                    getForegroundMask();
                }
                return BackgroundSegmentation::getBackgroundMask();
            }

            void postProcess() override {
                for (const auto &filter : filters) {
                    filter->apply(foregroundMask_gpu, foregroundMask_gpu, stream);
                }
            }
        };


        class MOG2 : public CudaBackgroundSegmentation {

        public:
            /** @brief Creates MOG2 Background Subtractor
             *
             * @param history Length of the history.
             * @param varThreshold Threshold on the squared Mahalanobis distance between the pixel and the model
             * to decide whether a pixel is well described by the background model. This parameter does not
             * affect the background update.
             * @param detectShadows If true, the algorithm will detect shadows and mark them. It decreases the
             * speed a bit, so if you do not need this feature, set the parameter to false.
             */
            explicit MOG2(int history = 500, double varThreshold = 16,
                          bool detectShadows = false) : CudaBackgroundSegmentation(
                    cv::cuda::createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)) {
                filters.emplace_back(
                        cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1,
                                                         cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                                                   cv::Size(5, 5)))
                );
                filters.emplace_back(
                        cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1,
                                                         cv::getStructuringElement(cv::MORPH_RECT,
                                                                                   cv::Size(7, 7)), cv::Point(-1, -1),
                                                         5)
                );
            }


            void applyGpu(const cv::cuda::GpuMat &frame) override {
                std::static_pointer_cast<cv::cuda::BackgroundSubtractorMOG2>(algorithm)->apply(frame,
                                                                                               foregroundMask_gpu,
                                                                                               -1, stream);
            };
        };

    }
}

#endif //DYNAMICSTABILIZATION_BACKGROUNDSEGMENTATION_H
