#include "opencv2/imgproc/imgproc.hpp"

#include "DynamicStabilization.hpp"
#include <opencv2/cudaarithm.hpp>

namespace providentia {
	namespace stabilization {

#pragma region DynamicStabilizerBase

		void DynamicStabilizerBase::stabilize(const cv::cuda::GpuMat &frame) {
			clear();
			updateKeyframe();

			frameFeatureDetector->detect(frame, segmentor->getBackgroundMask(frame.size()));
			if (referenceFeatureDetector->isEmpty()) {
				// TODO copy/clone frameFeatureDetector
				referenceFeatureDetector->detect(frame, segmentor->getBackgroundMask(frame.size()));
			}

			matcher->match(frameFeatureDetector, referenceFeatureDetector);
			warper->warp(frame, matcher);
			segmentor->segment(getStabilizedFrame());
			addTimestamp("Stabilization finished", 0);
		}

		void DynamicStabilizerBase::updateKeyframe() {
			if (!shouldUpdateKeyframe || currentIteration++ < warmUp) {
				return;
			}

			int newCount = cv::cuda::countNonZero(segmentor->getBackgroundMask(getStabilizedFrame().size()));
			int oldCount = cv::cuda::countNonZero(referenceFeatureDetector->getCurrentMask());
			if (oldCount == getStabilizedFrame().size().width * getStabilizedFrame().size().height ||
				newCount > oldCount) {
				referenceFeatureDetector->detect(getStabilizedFrame(),
												 segmentor->getBackgroundMask(
														 getStabilizedFrame().size())
													 .clone());
			}
		}

		cv::Mat DynamicStabilizerBase::getHomography() const {
			return warper->getHomography();
		}

		const cv::cuda::GpuMat &DynamicStabilizerBase::getStabilizedFrame() const {
			return warper->getWarpedFrame();
		}

		const cv::cuda::GpuMat &DynamicStabilizerBase::getOriginalFrame() const {
			return frameFeatureDetector->getOriginalFrame();
		}

		DynamicStabilizerBase::DynamicStabilizerBase() : providentia::utils::TimeMeasurable(
			"DynamicStabilizerBase", 1) {
			warper = std::make_shared<FrameWarper>();
			segmentor = std::make_shared<providentia::stabilization::segmentation::MOG2BackgroundSegmention>(
				cv::Size(1920, 1200) /
				10);
		}

		cv::Mat DynamicStabilizerBase::draw() {
			cv::Mat result;
			cv::hconcat(std::vector<cv::Mat>{cv::Mat(frameFeatureDetector->getOriginalFrame()),
											 cv::Mat(getStabilizedFrame())},
						result);
			return result;
		}

		const cv::cuda::GpuMat &DynamicStabilizerBase::getReferenceframe() const {
			return referenceFeatureDetector->getOriginalFrame();
		}

		const cv::cuda::GpuMat &DynamicStabilizerBase::getReferenceframeMask() const {
			return referenceFeatureDetector->getCurrentMask();
		}

		const std::shared_ptr<providentia::features::FeatureDetectorBase> &
		DynamicStabilizerBase::getFrameFeatureDetector() const {
			return frameFeatureDetector;
		}

		const std::shared_ptr<providentia::features::FeatureMatcherBase> &
		DynamicStabilizerBase::getMatcher() const {
			return matcher;
		}

		const std::shared_ptr<providentia::stabilization::segmentation::BackgroundSegmentionBase> &
		DynamicStabilizerBase::getSegmentor() const {
			return segmentor;
		}

		cv::cuda::GpuMat DynamicStabilizerBase::getBackgroundMask(const cv::Size &size) const {
			return segmentor->getBackgroundMask(size);
		}

		const std::shared_ptr<FrameWarper> &
		DynamicStabilizerBase::getWarper() const {
			return warper;
		}

		bool DynamicStabilizerBase::isShouldUpdateKeyframe() const {
			return shouldUpdateKeyframe;
		}

		void DynamicStabilizerBase::setShouldUpdateKeyframe(bool value) {
			shouldUpdateKeyframe = value;
		}

		void DynamicStabilizerBase::setShouldUseFundamentalMatrix(bool shouldUseFundamentalMatrix) {
			matcher->setShouldUseFundamentalMatrix(shouldUseFundamentalMatrix);
		}

		void DynamicStabilizerBase::setSkewThreshold(double skewThreshold) {
			warper->setSkewThreshold(skewThreshold);
		}

#pragma endregion DynamicStabilizerBase

#pragma region SURFBFDynamicStabilizer

		SURFBFDynamicStabilizer::SURFBFDynamicStabilizer(double hessianThreshold,
														 int nOctaves,
														 int nOctaveLayers,
														 bool extended,
														 float keypointsRatio,
														 bool upright) {
			auto detector = providentia::features::SURFFeatureDetector(hessianThreshold, nOctaves,
																	   nOctaveLayers, extended,
																	   keypointsRatio, upright);
			frameFeatureDetector = std::make_shared<providentia::features::SURFFeatureDetector>(detector);
			referenceFeatureDetector = std::make_shared<providentia::features::SURFFeatureDetector>(detector);
			matcher = std::make_shared<providentia::features::BruteForceFeatureMatcher>(cv::NORM_L2);
			setName(typeid(*this).name());
		}

#pragma endregion SURFBFDynamicStabilizer

#pragma region ORBBFDynamicStabilizer

		ORBBFDynamicStabilizer::ORBBFDynamicStabilizer(int nfeatures, float scaleFactor,
													   int nlevels, int edgeThreshold,
													   int firstLevel, int wtaK,
													   int scoreType,
													   int patchSize, int fastThreshold,
													   bool blurForDescriptor) {
			auto detector = providentia::features::ORBFeatureDetector(nfeatures, scaleFactor,
																	  nlevels, edgeThreshold,
																	  firstLevel, wtaK, scoreType,
																	  patchSize, fastThreshold,
																	  blurForDescriptor);
			frameFeatureDetector = std::make_shared<providentia::features::ORBFeatureDetector>(detector);
			referenceFeatureDetector = std::make_shared<providentia::features::ORBFeatureDetector>(detector);
			matcher = std::make_shared<providentia::features::BruteForceFeatureMatcher>(cv::NORM_HAMMING);
			setName(typeid(*this).name());
		}

#pragma endregion ORBBFDynamicStabilizer

		FastFREAKBFDynamicStabilizer::FastFREAKBFDynamicStabilizer(int threshold,
																   bool nonmaxSuppression,
																   cv::FastFeatureDetector::DetectorType type,
																   int maxNPoints,
																   bool orientationNormalized,
																   bool scaleNormalized,
																   float patternScale,
																   int nOctaves,
																   const std::vector<int> &selectedPairs) {
			auto detector = providentia::features::FastFREAKFeatureDetector(threshold,
																			nonmaxSuppression, type,
																			maxNPoints,
																			orientationNormalized,
																			scaleNormalized,
																			patternScale, nOctaves,
																			selectedPairs);
			frameFeatureDetector = std::make_shared<providentia::features::FastFREAKFeatureDetector>(detector);
			referenceFeatureDetector = std::make_shared<providentia::features::FastFREAKFeatureDetector>(detector);
			matcher = std::make_shared<providentia::features::BruteForceFeatureMatcher>(cv::NORM_HAMMING);
			setName(typeid(*this).name());
		}

	}// namespace stabilization
}// namespace providentia
