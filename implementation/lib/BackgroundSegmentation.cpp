#include <opencv2/cudawarping.hpp>
#include <utility>
#include "BackgroundSegmentation.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudabgsegm.hpp"

namespace providentia {
	namespace segmentation {

#pragma region BackgroundSegmentorBase

		void BackgroundSegmentionBase::segment(const cv::cuda::GpuMat &frame) {
			clear();

			if (calculationSize.empty()) {
				calculationSize = frame.size();
			}

			if (calculationSize != frame.size()) {
				cv::cuda::resize(frame, calculationFrame, calculationSize);
			} else {
				calculationFrame = frame.clone();
			}

			specificApply();
			if (all255Mask.empty() || all255Mask.size() != foregroundMask.size()) {
				all255Mask.upload(cv::Mat::ones(foregroundMask.size(), CV_8UC1) * 255);
			}

			postProcess();
			cv::cuda::absdiff(all255Mask, foregroundMask, backgroundMask, stream);
			stream.waitForCompletion();
			addTimestamp("Background segmentation finished", 0);
		}

		void BackgroundSegmentionBase::postProcess() {
			for (const auto &filter : filters) {
				filter->apply(foregroundMask, foregroundMask, stream);
			}
		}

		const cv::cuda::GpuMat &
		BackgroundSegmentionBase::getForegroundMask(const cv::Size &size) {
			if (foregroundMask.empty() ||
				cv::cuda::countNonZero(foregroundMask) == foregroundMask.size().width * foregroundMask.size().height) {
				clearMasks(size);
			}
			return foregroundMask;
		}

		const cv::cuda::GpuMat &
		BackgroundSegmentionBase::getBackgroundMask(const cv::Size &size) {
			if (backgroundMask.empty() ||
				cv::cuda::countNonZero(backgroundMask) == 0) {
				clearMasks(size);
			}
			if (!size.empty() && backgroundMask.size() != size) {
				cv::cuda::resize(backgroundMask, backgroundMask, size);
			}
			return backgroundMask;
		}

		cv::Mat BackgroundSegmentionBase::draw() const {
			cv::Mat result;
			cv::hconcat(std::vector<cv::Mat>{cv::Mat(backgroundMask), cv::Mat(foregroundMask)}, result);
			cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
			return result;
		}

		void BackgroundSegmentionBase::clearMasks(const cv::Size &size) {
			foregroundMask.upload(cv::Mat::ones(size, CV_8UC1) * 0);
			backgroundMask.upload(cv::Mat::ones(size, CV_8UC1) * 255);
		}

		BackgroundSegmentionBase::BackgroundSegmentionBase(cv::Size size)
			: providentia::utils::TimeMeasurable(
			"BackgroundSegmentionBase", 1), calculationSize(std::move(size)) {
		}

		void BackgroundSegmentionBase::addFilters() {
			// Optional filters can be added by overriding this function.
		}

#pragma endregion BackgroundSegmentorBase

#pragma region MOG2BackgroundSegmentor

		MOG2BackgroundSegmention::MOG2BackgroundSegmention(cv::Size calculationSize,
														 int history,
														 double varThreshold,
														 bool detectShadows)
			: BackgroundSegmentionBase(std::move(calculationSize)) {
			algorithm = cv::cuda::createBackgroundSubtractorMOG2(history, varThreshold, detectShadows);
			addFilters();
		}

		void MOG2BackgroundSegmention::specificApply() {
			algorithm->apply(calculationFrame, foregroundMask, -1, stream);
		}

		void MOG2BackgroundSegmention::addFilters() {
			BackgroundSegmentionBase::addFilters();

//    filters.emplace_back(cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1,
//                                                          cv::getStructuringElement(cv::MORPH_RECT,
//                                                                                    cv::Size(3, 3))));
//    filters.emplace_back(
//            cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1,
//                                             cv::getStructuringElement(cv::MORPH_RECT,
//                                                                       cv::Size(5, 5)), cv::Point(-1, -1), 3
//            ));
			filters.emplace_back(
				cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1,
												 cv::getStructuringElement(cv::MORPH_RECT,
																		   cv::Size(3, 3))
				));
//    filters.emplace_back(
//            cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1,
//                                             cv::getStructuringElement(cv::MORPH_RECT,
//                                                                       cv::Size(3, 3)), cv::Point(-1, -1), 5
//            ));
			filters.emplace_back(
				cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1,
												 cv::getStructuringElement(cv::MORPH_RECT,
																		   cv::Size(5, 5)), cv::Point(-1, -1), 3
				));
		}

#pragma endregion MOG2BackgroundSegmentor
	}
}
