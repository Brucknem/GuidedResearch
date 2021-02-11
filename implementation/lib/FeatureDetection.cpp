//
// Created by brucknem on 12.01.21.
//
#include <opencv2/cudaimgproc.hpp>
#include "opencv2/features2d.hpp"
#include <utility>
#include "FeatureDetection.hpp"

namespace providentia {
	namespace features {

#pragma region Getters_Setters{

		void FeatureDetectorBase::setCurrentMask(cv::Size _size) {
			cv::Size size = std::move(_size);
			if (size.empty()) {
				size = frame.size();
			}
			if (useLatestMask) {
				if (latestMask.empty()) {
					latestMask.upload(cv::Mat::ones(size, CV_8UC1) * 255);
				}
				currentMask = latestMask;
			} else {
				if (noMask.empty() || noMask.size() != size) {
					noMask.upload(cv::Mat::ones(size, CV_8UC1) * 255);
				}
				currentMask = noMask;
			}
		}

#pragma endregion Getters_Setters

#pragma region FeatureDetectorBase

		void FeatureDetectorBase::detect(const cv::cuda::GpuMat &_frame) {
			clear();
			originalFrame = _frame.clone();
			cv::cuda::cvtColor(_frame, frame, cv::COLOR_BGR2GRAY);
			frame.download(frameCPU);
			setCurrentMask();
			specificDetect();
			addTimestamp("Detection finished", 0);
		}

		void FeatureDetectorBase::detect(const cv::cuda::GpuMat &_frame, bool _useLatestMask) {
			useLatestMask = _useLatestMask;
			detect(_frame);
		}

		void FeatureDetectorBase::detect(const cv::cuda::GpuMat &_frame,
										 const cv::cuda::GpuMat &_mask) {
			latestMask = _mask;
			detect(_frame, true);
		}

		const cv::cuda::GpuMat &FeatureDetectorBase::getOriginalFrame() const {
			return originalFrame;
		}

		FeatureDetectorBase::FeatureDetectorBase() : providentia::utils::TimeMeasurable(
			"FeatureDetectorBase", 1) {}

		const cv::cuda::GpuMat &FeatureDetectorBase::getCurrentMask(cv::Size _size) {
			setCurrentMask(std::move(_size));
			return currentMask;
		}

		const std::vector<cv::KeyPoint> &FeatureDetectorBase::getKeypoints() const {
			return keypointsCPU;
		}

		bool FeatureDetectorBase::isEmpty() {
			return keypointsCPU.empty();
		}

		cv::Mat FeatureDetectorBase::draw() {
			cv::drawKeypoints(cv::Mat(originalFrame), keypointsCPU, drawFrame);
			return drawFrame;
		}

		const cv::cuda::GpuMat &FeatureDetectorBase::getDescriptorsGPU() const {
			return descriptorsGPU;
		}

		const cv::Mat &FeatureDetectorBase::getDescriptorsCPU() const {
			return descriptorsCPU;
		}

#pragma endregion FeatureDetectorBase

#pragma region SURFFeatureDetector

		SURFFeatureDetector::SURFFeatureDetector(double _hessianThreshold, int _nOctaves,
												 int _nOctaveLayers, bool _extended,
												 float _keypointsRatio, bool _upright) {
			detector = cv::cuda::SURF_CUDA::create(_hessianThreshold, _nOctaves, _nOctaveLayers, _extended,
												   _keypointsRatio, _upright);
			setName(typeid(*this).name());
		}

		void SURFFeatureDetector::specificDetect() {
			detector->detectWithDescriptors(frame, getCurrentMask(frame.size()), keypointsGPU, descriptorsGPU, false);
			detector->downloadKeypoints(keypointsGPU, keypointsCPU);
			descriptorsGPU.download(descriptorsCPU);
		}

#pragma endregion SURFFeatureDetector

#pragma region ORBFeatureDetector

		ORBFeatureDetector::ORBFeatureDetector(int nfeatures, float scaleFactor, int nlevels,
											   int edgeThreshold, int firstLevel, int WTA_K,
											   int scoreType, int patchSize, int fastThreshold,
											   bool blurForDescriptor) {
			detector = cv::cuda::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K,
											 scoreType,
											 patchSize, fastThreshold, blurForDescriptor);
			setName(typeid(*this).name());
		}

		void ORBFeatureDetector::specificDetect() {
			detector->detectAndComputeAsync(frame, getCurrentMask(frame.size()), keypointsGPU, descriptorsGPU, false);
			detector->convert(keypointsGPU, keypointsCPU);
			descriptorsGPU.download(descriptorsCPU);
		}

#pragma endregion ORBFeatureDetector

#pragma region FastFREAKFeatureDetector

		FastFREAKFeatureDetector::FastFREAKFeatureDetector(int threshold, bool nonmaxSuppression,
														   cv::FastFeatureDetector::DetectorType type,
														   int max_npoints,
														   bool orientationNormalized,
														   bool scaleNormalized,
														   float patternScale,
														   int nOctaves,
														   const std::vector<int> &selectedPairs) {
			detector = cv::cuda::FastFeatureDetector::create(threshold, nonmaxSuppression, type, max_npoints);
			descriptor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves,
														selectedPairs);
			setName(typeid(*this).name());
		}

		void FastFREAKFeatureDetector::specificDetect() {
			detector->detect(frame, keypointsCPU, getCurrentMask(frame.size()));
			descriptor->compute(frameCPU, keypointsCPU, descriptorsCPU);
			descriptorsGPU.upload(descriptorsCPU);
		}

#pragma endregion FastFREAKFeatureDetector

#pragma region SIFTFeatureDetector

		SIFTFeatureDetector::SIFTFeatureDetector(int nfeatures, int nOctaveLayers,
												 double contrastThreshold, double edgeThreshold,
												 double sigma) {
			detector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
			setName(typeid(*this).name());
		}

		void SIFTFeatureDetector::specificDetect() {
			detector->detectAndCompute(frameCPU, cv::Mat(getCurrentMask()), keypointsCPU, descriptorsCPU);
			descriptorsGPU.upload(descriptorsCPU);
		}

#pragma endregion SIFTFeatureDetector

#pragma region StarBRIEFFeatureDetector

		StarBRIEFFeatureDetector::StarBRIEFFeatureDetector(int maxSize, int responseThreshold,
														   int lineThresholdProjected,
														   int lineThresholdBinarized,
														   int suppressNonmaxSize, int bytes,
														   bool use_orientation) {
			detector = cv::xfeatures2d::StarDetector::create(maxSize, responseThreshold, lineThresholdProjected,
															 lineThresholdBinarized, suppressNonmaxSize);
			descriptor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
			setName(typeid(*this).name());
		}

		void StarBRIEFFeatureDetector::specificDetect() {
			detector->detect(frameCPU, keypointsCPU, cv::Mat(getCurrentMask(frame.size())));
			descriptor->compute(frameCPU, keypointsCPU, descriptorsCPU);
			descriptorsGPU.upload(descriptorsCPU);
		}

#pragma endregion StarBRIEFFeatureDetector

	}
}