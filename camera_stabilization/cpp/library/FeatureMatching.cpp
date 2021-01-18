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

            //-- Filter matches using the Lowe's ratio test
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
            addTimestamp("Matching finished", 0);
        }

        cv::Mat FeatureMatcherBase::draw() {
            drawMatches(
                    cv::Mat(frameDetector->getOriginalFrame()), frameDetector->getKeypoints(),
                    cv::Mat(referenceFrameDetector->getOriginalFrame()), referenceFrameDetector->getKeypoints(),
                    goodMatches, drawFrame, cv::Scalar::all(-1),
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

        const std::vector<cv::Point2f> &FeatureMatcherBase::getReferenceMatchedPoints() const {
            return referenceMatchedPoints;
        }

        FeatureMatcherBase::FeatureMatcherBase(float _goodMatchRatioThreshold)
                : providentia::utils::TimeMeasurable("FeatureMatcherBase", 1), goodMatchRatioThreshold(
                _goodMatchRatioThreshold) {
        }

        FeatureMatcherBase::~FeatureMatcherBase() = default;

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

        BruteForceFeatureMatcher::~BruteForceFeatureMatcher() = default;

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
                                                 float _goodMatchRatioThreshold) {
            if (binaryDescriptors) {
                matcher = std::make_shared<cv::FlannBasedMatcher>(new cv::flann::LshIndexParams(12, 20, 2));
            } else {
                matcher = std::make_shared<cv::FlannBasedMatcher>();
            }
        }

        FlannFeatureMatcher::~FlannFeatureMatcher() = default;

#pragma endregion FlannFeatureMatching
    }
}
