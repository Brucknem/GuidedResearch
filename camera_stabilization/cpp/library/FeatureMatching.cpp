//
// Created by brucknem on 12.01.21.
//

#include "FeatureMatching.hpp"

#pragma region FeatureMatchingBase

void providentia::features::FeatureMatcherBase::match(const std::shared_ptr<FeatureDetectorBase> &_frameFeatureDetector,
                                                      const std::shared_ptr<FeatureDetectorBase> &_referenceFeatureDetector) {
    frameFeatureDetector = _frameFeatureDetector;
    referenceFeatureDetector = _referenceFeatureDetector;

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
    auto keypointsFrameCPU = frameFeatureDetector->getKeypointsCPU();
    auto keypointsReferenceFrameCPU = referenceFeatureDetector->getKeypointsCPU();
    for (auto &goodMatch : goodMatches) {
        //-- Get the keypointsGPU from the good matches
        frameMatchedPoints.push_back(keypointsFrameCPU[goodMatch.queryIdx].pt);
        referenceMatchedPoints.push_back(keypointsReferenceFrameCPU[goodMatch.trainIdx].pt);
    }
}

cv::Mat providentia::features::FeatureMatcherBase::draw() {
    drawMatches(cv::Mat(frameFeatureDetector->getOriginalFrame()), frameFeatureDetector->getKeypointsCPU(),
                cv::Mat(referenceFeatureDetector->getOriginalFrame()), referenceFeatureDetector->getKeypointsCPU(),
                goodMatches, drawFrame, cv::Scalar::all(-1),
                cv::Scalar::all(-1), std::vector<char>(),
                cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return drawFrame;
}

const std::vector<cv::DMatch> &providentia::features::FeatureMatcherBase::getGoodMatches() const {
    return goodMatches;
}

const std::vector<cv::Point2f> &providentia::features::FeatureMatcherBase::getFrameMatchedPoints() const {
    return frameMatchedPoints;
}

const std::vector<cv::Point2f> &providentia::features::FeatureMatcherBase::getReferenceMatchedPoints() const {
    return referenceMatchedPoints;
}

#pragma endregion FeatureMatchingBase

#pragma region BruteForceFeatureMatching

providentia::features::BruteForceFeatureMatcher::BruteForceFeatureMatcher(cv::NormTypes norm,
                                                                          float _goodMatchRatioThreshold)
        : FeatureMatcherBase(_goodMatchRatioThreshold) {
    matcher = cv::cuda::DescriptorMatcher::createBFMatcher(norm);
}


void providentia::features::BruteForceFeatureMatcher::specificMatch() {
    matcher->knnMatchAsync(frameFeatureDetector->getDescriptorsGPU(), referenceFeatureDetector->getDescriptorsGPU(),
                           knnMatchesGPU, 2, cv::noArray(), stream);
    stream.waitForCompletion();
    matcher->knnMatchConvert(knnMatchesGPU, knnMatchesCPU);
}


#pragma endregion BruteForceFeatureMatching

