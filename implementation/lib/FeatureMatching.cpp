//
// Created by brucknem on 12.01.21.
//

#include "FeatureMatching.hpp"

namespace providentia {
	namespace features {

#pragma region FeatureMatchingBase

		void FeatureMatcherBase::match(
			const std::shared_ptr<FeatureDetectorBase> &_frameFeatureDetector,
			const std::shared_ptr<FeatureDetectorBase> &_referenceFeatureDetector
		) {
			clear();
			frameDetector = _frameFeatureDetector;
			referenceFrameDetector = _referenceFeatureDetector;

			if (frameDetector->getKeypoints().empty() || referenceFrameDetector->getKeypoints().empty()) {
				addTimestamp("Matching finished", 0);
				return;
			}

			specificMatch();

			//-- Filter matches using the Lowe's ratio toCameraSpace
			goodMatches.clear();

			for (auto &knnMatchCPU : knnMatchesCPU) {
				if (knnMatchCPU[0].distance < goodMatchRatioThreshold * knnMatchCPU[1].distance) {
					goodMatches.push_back(knnMatchCPU[0]);
				}
			}

			//-- Localize the object
			frameMatchedPoints.clear();
			referenceMatchedPoints.clear();

			for (auto &goodMatch : goodMatches) {
				//-- Get the keypoints from the good matches
				frameMatchedPoints.push_back(frameDetector->getKeypoints()[goodMatch.queryIdx].pt);
				referenceMatchedPoints.push_back(referenceFrameDetector->getKeypoints()[goodMatch.trainIdx].pt);
			}

			if (shouldUseFundamentalMatrix) {
				filterUsingFundamentalMatrix();
			} else {
				fundamentalMatches = goodMatches;
			}

			addTimestamp("Matching finished", 0);
		}

		void FeatureMatcherBase::filterUsingFundamentalMatrix() {
			fundamentalMatrix = cv::findFundamentalMat(frameMatchedPoints, referenceMatchedPoints,
													   cv::FM_RANSAC, 1.0, 0.975, fundamentalMatrixInlierMask);
			const cv::Mat &essentialMatrix = cv::findFundamentalMat(frameMatchedPoints, referenceMatchedPoints,
																	cv::FM_RANSAC, 1.0, 0.975,
																	fundamentalMatrixInlierMask);
			fundamentalMatches.clear();
			for (int i = 0; i < goodMatches.size(); i++) {
				if (fundamentalMatrixInlierMask.at<bool>(i, 0)) {
					fundamentalMatches.push_back(goodMatches[i]);
				}
			}

			//-- Localize the object
			frameMatchedPoints.clear();
			referenceMatchedPoints.clear();
			for (auto &goodMatch : fundamentalMatches) {
				//-- Get the keypoints from the good matches
				frameMatchedPoints.push_back(frameDetector->getKeypoints()[goodMatch.queryIdx].pt);
				referenceMatchedPoints.push_back(referenceFrameDetector->getKeypoints()[goodMatch.trainIdx].pt);
			}
		}

		cv::Mat FeatureMatcherBase::draw() {
			drawMatches(
				cv::Mat(frameDetector->getOriginalFrame()), frameDetector->getKeypoints(),
				cv::Mat(referenceFrameDetector->getOriginalFrame()), referenceFrameDetector->getKeypoints(),
				fundamentalMatches, drawFrame, cv::Scalar::all(-1),
				cv::Scalar::all(-1), std::vector<char>(),
				cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			return drawFrame;
		}

		const std::vector<cv::DMatch> &FeatureMatcherBase::getGoodMatches() const {
			return goodMatches;
		}

		const std::vector<cv::Point2f> &FeatureMatcherBase::getFrameMatchedPoints() const {
			return frameMatchedPoints;
		}

		const std::vector<cv::Point2f> &FeatureMatcherBase::getReferenceMatchedPoints() {
			return referenceMatchedPoints;
		}

		FeatureMatcherBase::FeatureMatcherBase(float _goodMatchRatioThreshold)
			: providentia::utils::TimeMeasurable("FeatureMatcherBase", 1), goodMatchRatioThreshold(
			_goodMatchRatioThreshold) {
		}

		void FeatureMatcherBase::setShouldUseFundamentalMatrix(bool _shouldUseFundamentalMatrix) {
			shouldUseFundamentalMatrix = _shouldUseFundamentalMatrix;
		}

#pragma endregion FeatureMatchingBase

#pragma region BruteForceFeatureMatching

		BruteForceFeatureMatcher::BruteForceFeatureMatcher(cv::NormTypes norm,
														   float _goodMatchRatioThreshold)
			: FeatureMatcherBase(_goodMatchRatioThreshold) {
			matcher = cv::cuda::DescriptorMatcher::createBFMatcher(norm);
			setName(typeid(*this).name());
		}

		void BruteForceFeatureMatcher::specificMatch() {
			if (frameDetector->getDescriptorsGPU().empty() || frameDetector->getDescriptorsGPU().empty()) {
				throw std::invalid_argument("Possibly match with wrong descriptor format called.");
			}
			matcher->knnMatchAsync(frameDetector->getDescriptorsGPU(), referenceFrameDetector->getDescriptorsGPU(),
								   knnMatchesGPU, 2, cv::noArray(), stream);
			stream.waitForCompletion();
			matcher->knnMatchConvert(knnMatchesGPU, knnMatchesCPU);
		}

#pragma endregion BruteForceFeatureMatching

#pragma region FlannFeatureMatching

		FlannFeatureMatcher::FlannFeatureMatcher(cv::flann::IndexParams *params,
												 float _goodMatchRatioThreshold)
			: FeatureMatcherBase(
			_goodMatchRatioThreshold) {
			matcher = std::make_shared<cv::FlannBasedMatcher>(params);
			setName(typeid(*this).name());
		}

		void FlannFeatureMatcher::specificMatch() {
			if (frameDetector->getDescriptorsCPU().empty() || referenceFrameDetector->getDescriptorsCPU().empty()) {
				throw std::invalid_argument("Possibly match with wrong descriptor format called.");
			}
			matcher->knnMatch(frameDetector->getDescriptorsCPU(), referenceFrameDetector->getDescriptorsCPU(),
							  knnMatchesCPU,
							  2);
		}

		FlannFeatureMatcher::FlannFeatureMatcher(bool binaryDescriptors,
												 float _goodMatchRatioThreshold) : FeatureMatcherBase(
			_goodMatchRatioThreshold) {
			if (binaryDescriptors) {
				matcher = std::make_shared<cv::FlannBasedMatcher>(new cv::flann::LshIndexParams(12, 20, 2));
			} else {
				matcher = std::make_shared<cv::FlannBasedMatcher>();
			}
		}

#pragma endregion FlannFeatureMatching
	}
}
