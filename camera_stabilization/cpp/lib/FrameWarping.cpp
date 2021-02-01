//
// Created by brucknem on 13.01.21.
//

#include <opencv2/cudawarping.hpp>
#include "FrameWarping.hpp"

namespace providentia {
	namespace stabilization {

		void FrameWarper::warp(const cv::cuda::GpuMat &_frame,
							   const std::shared_ptr<providentia::features::FeatureMatcherBase> &matcher) {
			clear();
			if (matcher->getFrameMatchedPoints().size() < 4) {
				homography = cv::Mat::eye(3, 3, CV_64F);
			} else {
				homography = cv::findHomography(matcher->getFrameMatchedPoints(), matcher->getReferenceMatchedPoints(),
												homographyCalculationAlgorithm);
			}

			cv::cuda::warpPerspective(_frame, warpedFrame, homography, _frame.size(), perspectiveWarpFlags);
			addTimestamp("Frame warping finished", 0);
		}

		providentia::stabilization::FrameWarper::FrameWarper() : providentia::utils::TimeMeasurable("Frame Warper", 1) {

		}

		const cv::cuda::GpuMat &providentia::stabilization::FrameWarper::getWarpedFrame() const {
			return warpedFrame;
		}

		const cv::Mat &providentia::stabilization::FrameWarper::getHomography() const {
			return homography;
		}

	}
}
