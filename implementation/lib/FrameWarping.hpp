//
// Created by brucknem on 13.01.21.
//

#ifndef CAMERASTABILIZATION_FRAMEWARPING_HPP
#define CAMERASTABILIZATION_FRAMEWARPING_HPP

#include "opencv2/opencv.hpp"
#include "FeatureMatching.hpp"
#include "TimeMeasurable.hpp"

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
			 * Minimizer for the reprojection error between the writeFrames.
			 */
			cv::Mat homography;

			/**
			 * The current frame warpedFrame by the found homography and with minimal reprojection error to the reference frame.
			 */
			cv::cuda::GpuMat warpedFrame;

			double skewThreshold = 1e-4;

		public:

			/**
			 * @destructor
			 */
			~FrameWarper() override = default;

			/**
			 * @get The final frame warped by the found homography.
			 */
			const cv::cuda::GpuMat &getWarpedFrame() const;

			/**
			 * @get The homography that minimizes the reprojection error.
			 */
			cv::Mat getHomography(double _skewThreshold = 1e-4) const;

			double getMaxSkew() const;

			void setSkewThreshold(double _skewThreshold);

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
			warp(const cv::cuda::GpuMat &_frame,
				 const std::shared_ptr<providentia::features::FeatureMatcherBase> &matcher);

			static cv::cuda::GpuMat warp(const cv::cuda::GpuMat &_frame, const cv::Mat &homography, int
			perspectiveWarpFlags = cv::INTER_LINEAR);
		};
	}
}

#endif //CAMERASTABILIZATION_FRAMEWARPING_HPP
