//
// Created by brucknem on 13.01.21.
//

#ifndef CAMERASTABILIZATION_BACKGROUNDSEGMENTATION_HPP
#define CAMERASTABILIZATION_BACKGROUNDSEGMENTATION_HPP

#include <opencv2/cudabgsegm.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/cudafilters.hpp"

namespace providentia {
    namespace segmentation {
        class BackgroundSegmentorBase {
        private:

            /**
             * An internal used mask consisting of on 255 values used for
             * calculating the background mask as the difference of 255 - foreground mask.
             */
            cv::cuda::GpuMat all255Mask;

        protected:
            cv::cuda::Stream stream;

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
            virtual void specificApply(const cv::cuda::GpuMat &_frame) = 0;

            /**
             * Additional postprocessing steps on the raw background segmentation result of the algorithm.
             */
            virtual void postProcess();

        public:
            /**
             * Appends the given frame to the internal history of frames and calculates the background segmentation.
             *
             * @param _frame The new frame to apply.
             */
            void apply(const cv::cuda::GpuMat &_frame);

            /**
             * @get
             *
             * @return The background mask.
             */
            const cv::cuda::GpuMat &getBackgroundMask() const;

            /**
             * @get
             *
             * @return The foreground mask.
             */
            const cv::cuda::GpuMat &getForegroundMask() const;
        };

        class MOG2BackgroundSegmentor : public BackgroundSegmentorBase {
        private:
            cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> algorithm;

        protected:
            void specificApply(const cv::cuda::GpuMat &_frame) override;

        public:
            explicit MOG2BackgroundSegmentor(int history = 500, double varThreshold = 16,
                                             bool detectShadows = false);
        };
    }
}


#endif //CAMERASTABILIZATION_BACKGROUNDSEGMENTATION_HPP
