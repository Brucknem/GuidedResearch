//
// Created by brucknem on 13.01.21.
//

#ifndef CAMERASTABILIZATION_FRAMEWARPING_H
#define CAMERASTABILIZATION_FRAMEWARPING_H

#include "opencv2/opencv.hpp"
#include "FeatureMatching.hpp"
#include "Utils.hpp"

namespace providentia {
    namespace stabilization {
        /**
         * Base class for the frame warpers.
         */
        class FrameWarper : public providentia::utils::TimeMeasurable {
        protected:

            /**
             * The algorithm used to find the homography.
             */
            int homographyCalculationAlgorithm = cv::RANSAC;

            /**
             * The interpolation algorithm used to warp the frame.
             */
            int perspectiveWarpFlags = cv::INTER_LINEAR;

            /**
             * The found homography between the matched keypoints.
             * Minimizer for the reprojection error between the frames.
             */
            cv::Mat homography;

            /**
             * The current frame warpedFrame by the found homography and with minimal reprojection error to the reference frame.
             */
            cv::cuda::GpuMat warpedFrame;

        public:
            /**
             * @get The final frame warped by the found homography.
             */
            const cv::cuda::GpuMat &getWarpedFrame() const;

            /**
             * @get The homography that minimizes the reprojection error.
             */
            const cv::Mat &getHomography() const;

            /**
             * @constructor
             */
            explicit FrameWarper();

            /**
             * Finds the homography that minimizes the reprojection error between the feature matches and warpes the frame.
             * @param _frame The frame to warp.
             * @param matcher The matched features.
             */
            void
            warp(const cv::cuda::GpuMat &_frame, std::shared_ptr<providentia::features::FeatureMatcherBase> matcher);

        };
    }
}


#endif //CAMERASTABILIZATION_FRAMEWARPING_H
