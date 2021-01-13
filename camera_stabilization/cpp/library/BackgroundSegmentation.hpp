//
// Created by brucknem on 13.01.21.
//

#ifndef CAMERASTABILIZATION_BACKGROUNDSEGMENTATION_HPP
#define CAMERASTABILIZATION_BACKGROUNDSEGMENTATION_HPP

#include <opencv2/cudabgsegm.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/cudafilters.hpp"
#include "Utils.hpp"

namespace providentia {
    namespace segmentation {
        class BackgroundSegmentorBase : public providentia::utils::TimeMeasurable {
        private:

            /**
             * An internal used mask consisting of on 255 values used for
             * calculating the background mask as the difference of 255 - foreground mask.
             */
            cv::cuda::GpuMat all255Mask;


        protected:
            /**
             * The size used for background calculation.
             */
            cv::Size calculationSize;
            /**
             * A CUDA stream used for async calculations.
             */
            cv::cuda::Stream stream;

            /**
             * The frame used during calculations.
             */
            cv::cuda::GpuMat frame;

            /**
             * The fore- and background masks.
             */
            cv::cuda::GpuMat foregroundMask, backgroundMask;

            /**
             * Some additional filters used during post processing.
             */
            std::vector<cv::Ptr<cv::cuda::Filter>> filters;

            /**
             * Subclass specifc algorithm implementation.
             *
             * @param _frame The new frame to apply the algorithm to.
             */
            virtual void specificApply() = 0;

            /**
             * Additional postprocessing steps on the raw background segmentation result of the algorithm.
             */
            void postProcess();

            /**
             * @constructor
             */
            explicit BackgroundSegmentorBase(cv::Size calculationSize = cv::Size());

            /**
             * Add optional filters to the postprocessing step.
             */
            virtual void addFilters();

        public:
            /**
             * Appends the given frame to the internal history of frames and calculates the background segmentation.
             *
             * @param _frame The new frame to apply.
             */
            virtual void segment(const cv::cuda::GpuMat &_frame);

            /**
             * Clears the masks to all background.
             */
            void clearMasks(const cv::Size &_size = cv::Size());

            /**
             * @get
             *
             * @return The background mask.
             */
            const cv::cuda::GpuMat &getBackgroundMask(const cv::Size &_size = cv::Size());

            /**
             * @get
             *
             * @return The foreground mask.
             */
            const cv::cuda::GpuMat &getForegroundMask(const cv::Size &_size = cv::Size());

            /**
             * Draws the foreground and background masks aside.
             */
            cv::Mat draw() const;
        };

        class MOG2BackgroundSegmentor : public BackgroundSegmentorBase {
        private:
            cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> algorithm;

        protected:
            void specificApply() override;

            void addFilters() override;

        public:
            explicit MOG2BackgroundSegmentor(cv::Size _calculationSize = cv::Size(), int history = 500,
                                             double varThreshold = 16,
                                             bool detectShadows = false);
        };
    }
}


#endif //CAMERASTABILIZATION_BACKGROUNDSEGMENTATION_HPP
