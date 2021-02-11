#include <opencv2/cudawarping.hpp>
#include <utility>
#include "BackgroundSegmentation.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudabgsegm.hpp"

namespace providentia {
	namespace segmentation {

#pragma region BackgroundSegmentorBase

		void BackgroundSegmentorBase::segment(const cv::cuda::GpuMat &_frame) {
			clear();

			if (calculationSize.empty()) {
				calculationSize = _frame.size();
			}

			if (calculationSize != _frame.size()) {
				cv::cuda::resize(_frame, frame, calculationSize);
			} else {
				frame = _frame.clone();
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

		void BackgroundSegmentorBase::postProcess() {
			for (const auto &filter : filters) {
				filter->apply(foregroundMask, foregroundMask, stream);
			}
		}

		const cv::cuda::GpuMat &
		BackgroundSegmentorBase::getForegroundMask(const cv::Size &_size) {
			if (foregroundMask.empty() ||
				cv::cuda::countNonZero(foregroundMask) == foregroundMask.size().width * foregroundMask.size().height) {
				clearMasks(_size);
			}
			return foregroundMask;
		}

		const cv::cuda::GpuMat &
		BackgroundSegmentorBase::getBackgroundMask(const cv::Size &_size) {
			if (backgroundMask.empty() ||
				cv::cuda::countNonZero(backgroundMask) == 0) {
				clearMasks(_size);
			}
			if (!_size.empty() && backgroundMask.size() != _size) {
				cv::cuda::resize(backgroundMask, backgroundMask, _size);
			}
			return backgroundMask;
		}

		cv::Mat BackgroundSegmentorBase::draw() const {
			cv::Mat result;
			cv::hconcat(std::vector<cv::Mat>{cv::Mat(backgroundMask), cv::Mat(foregroundMask)}, result);
			cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
			return result;
		}

		void BackgroundSegmentorBase::clearMasks(const cv::Size &_size) {
			foregroundMask.upload(cv::Mat::ones(_size, CV_8UC1) * 0);
			backgroundMask.upload(cv::Mat::ones(_size, CV_8UC1) * 255);
		}

		BackgroundSegmentorBase::BackgroundSegmentorBase(cv::Size _size)
			: providentia::utils::TimeMeasurable(
			"BackgroundSegmentorBase", 1), calculationSize(std::move(_size)) {
		}

		void BackgroundSegmentorBase::addFilters() {
			// Optional filters can be added by overriding this function.
		}

#pragma endregion BackgroundSegmentorBase

#pragma region MOG2BackgroundSegmentor

		MOG2BackgroundSegmentor::MOG2BackgroundSegmentor(cv::Size calculationSize,
														 int history,
														 double varThreshold,
														 bool detectShadows)
			: BackgroundSegmentorBase(std::move(calculationSize)) {
			algorithm = cv::cuda::createBackgroundSubtractorMOG2(history, varThreshold, detectShadows);
			addFilters();
		}

		void MOG2BackgroundSegmentor::specificApply() {
			algorithm->apply(frame, foregroundMask, -1, stream);
		}

		void MOG2BackgroundSegmentor::addFilters() {
			BackgroundSegmentorBase::addFilters();

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
