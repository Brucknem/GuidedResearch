//
// Created by brucknem on 12.01.21.
//

#include "FeatureMatching.hpp"

#include <utility>

#pragma region FeatureMatchingBase


void providentia::features::FeatureMatcherBase::match(
        const std::shared_ptr<providentia::features::FeatureDetectorBase> &_frameFeatureDetector,
        const std::shared_ptr<providentia::features::FeatureDetectorBase> &_referenceFeatureDetector
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
        //-- Get the keypointsGPU from the good matches
        frameMatchedPoints.push_back(frameDetector->getKeypoints()[goodMatch.queryIdx].pt);
        referenceMatchedPoints.push_back(referenceFrameDetector->getKeypoints()[goodMatch.trainIdx].pt);
    }
    addTimestamp("Matching finished", 0);
}

cv::Mat providentia::features::FeatureMatcherBase::draw() {
    drawMatches(
            cv::Mat(frameDetector->getOriginalFrame()), frameDetector->getKeypoints(),
            cv::Mat(referenceFrameDetector->getOriginalFrame()), referenceFrameDetector->getKeypoints(),
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

providentia::features::FeatureMatcherBase::FeatureMatcherBase(float _goodMatchRatioThreshold)
        : providentia::utils::TimeMeasurable("FeatureMatcherBase", 1), goodMatchRatioThreshold(
        _goodMatchRatioThreshold) {
}

#pragma endregion FeatureMatchingBase

#pragma region BruteForceFeatureMatching

providentia::features::BruteForceFeatureMatcher::BruteForceFeatureMatcher(cv::NormTypes norm,
                                                                          float _goodMatchRatioThreshold)
        : FeatureMatcherBase(_goodMatchRatioThreshold) {
    matcher = cv::cuda::DescriptorMatcher::createBFMatcher(norm);
    setName(typeid(*this).name());
}


void providentia::features::BruteForceFeatureMatcher::specificMatch() {
    if (frameDetector->getDescriptorsGPU().empty() || frameDetector->getDescriptorsGPU().empty()) {
        throw std::invalid_argument("Possibly match with wrong descriptor format called.");
    }
    matcher->knnMatchAsync(frameDetector->getDescriptorsGPU(), referenceFrameDetector->getDescriptorsGPU(),
                           knnMatchesGPU, 2, cv::noArray(), stream);
    stream.waitForCompletion();
    matcher->knnMatchConvert(knnMatchesGPU, knnMatchesCPU);
}


#pragma endregion BruteForceFeatureMatching

providentia::features::FlannFeatureMatcher::FlannFeatureMatcher(cv::flann::IndexParams *params,
                                                                float _goodMatchRatioThreshold) : FeatureMatcherBase(
        _goodMatchRatioThreshold) {
    matcher = std::make_shared<cv::FlannBasedMatcher>(params);
    setName(typeid(*this).name());
}

void providentia::features::FlannFeatureMatcher::specificMatch() {
    if (frameDetector->getDescriptorsCPU().empty() || referenceFrameDetector->getDescriptorsCPU().empty()) {
        throw std::invalid_argument("Possibly match with wrong descriptor format called.");
    }
    matcher->knnMatch(frameDetector->getDescriptorsCPU(), referenceFrameDetector->getDescriptorsCPU(), knnMatchesCPU,
                      2);
}

providentia::features::FlannFeatureMatcher::FlannFeatureMatcher(bool binaryDescriptors,
                                                                float _goodMatchRatioThreshold) {
    cv::flann::IndexParams *params;
    if (binaryDescriptors) {
        matcher = std::make_shared<cv::FlannBasedMatcher>(new cv::flann::LshIndexParams(20, 10, 2));
    } else {
        matcher = std::make_shared<cv::FlannBasedMatcher>();
    }
}
