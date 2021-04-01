//
// Created by brucknem on 13.01.21.
//

#include <opencv2/cudawarping.hpp>
#include "FrameWarping.hpp"

namespace providentia {
	namespace stabilization {

		cv::cuda::GpuMat FrameWarper::warp(const cv::cuda::GpuMat &_frame,
										   const cv::Mat &homography, int perspectiveWarpFlags) {
			if (homography.empty()) {
				return _frame;
			}
			cv::cuda::GpuMat result;
			cv::cuda::warpPerspective(_frame, result, homography, _frame.size(), perspectiveWarpFlags);
			return result;
		}

		void FrameWarper::warp(const cv::cuda::GpuMat &_frame,
							   const std::shared_ptr<providentia::features::FeatureMatcherBase> &matcher) {
			clear();
			if (matcher->getFrameMatchedPoints().size() < 4) {
				homography = cv::Mat::eye(3, 3, CV_64F);
			} else {
				homography = cv::findHomography(matcher->getFrameMatchedPoints(), matcher->getReferenceMatchedPoints(),
												homographyCalculationAlgorithm);
			}

//			cv::Vec2d translation = {homography.at<double>(0, 2), homography.at<double>(1, 2)};
//			cv::Vec4d rotation = {
//				homography.at<double>(0, 0), homography.at<double>(0, 1),
//				homography.at<double>(1, 0), homography.at<double>(1, 1)
//			};
			cv::Vec2d skew = {homography.at<double>(2, 0), homography.at<double>(2, 1)};

//			if (cv::norm(translation) > 10 ||
//				abs(sqrt(2) - cv::norm(rotation)) > 0.10 ||
			// TODO Evaluate skew threshold
			if (cv::norm(skew) > 1e-4) {
//				std::cout << "translation" << std::endl;
//				std::cout << translation << std::endl;
//				std::cout << cv::norm(translation) << std::endl;
//				std::cout << "rotation" << std::endl;
//				std::cout << rotation << std::endl;
//				std::cout << abs(sqrt(2) - cv::norm(rotation)) << std::endl;
//				std::cout << "skew" << std::endl;
//				std::cout << skew << std::endl;
//				std::cout << cv::norm(skew) << std::endl;
//				std::cout << homography << std::endl;

				homography = cv::Mat::eye(3, 3, CV_64F);
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
