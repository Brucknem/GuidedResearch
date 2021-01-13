#include "opencv2/imgproc/imgproc.hpp"

#include "DynamicStabilization.hpp"
#include <opencv2/cudawarping.hpp>

#pragma region DynamicStabilizerBase

void providentia::stabilization::DynamicStabilizerBase::stabilize(const cv::cuda::GpuMat &_frame) {
    frameFeatureDetector->detect(_frame);

    if (referenceFeatureDetector->isEmpty()) {
        // TODO copy/clone frameFeatureDetector
        referenceFeatureDetector->detect(_frame);
        homography = cv::Mat::eye(3, 3, CV_64F);
        stabilizedFrame = _frame.clone();
        return;
    }

    matcher->match(frameFeatureDetector, referenceFeatureDetector);

    if (matcher->getGoodMatches().size() < 4) {
        homography = cv::Mat::eye(3, 3, CV_64F);
    } else {
        homography = cv::findHomography(matcher->getFrameMatchedPoints(), matcher->getReferenceMatchedPoints(),
                                        cv::RANSAC);
    }

    cv::cuda::warpPerspective(_frame, stabilizedFrame, homography, _frame.size(), cv::INTER_LINEAR);
}

const cv::Mat &providentia::stabilization::DynamicStabilizerBase::getHomography() const {
    return homography;
}

const cv::cuda::GpuMat &providentia::stabilization::DynamicStabilizerBase::getStabilizedFrame() const {
    return stabilizedFrame;
}

#pragma endregion DynamicStabilizerBase

#pragma region SURFBFDynamicStabilizer

providentia::stabilization::SURFBFDynamicStabilizer::SURFBFDynamicStabilizer(double _hessianThreshold,
                                                                             int _nOctaves,
                                                                             int _nOctaveLayers,
                                                                             bool _extended,
                                                                             float _keypointsRatio,
                                                                             bool _upright) {
    frameFeatureDetector = std::make_shared<providentia::features::SurfFeatureDetector>(_hessianThreshold, _nOctaves,
                                                                                        _nOctaveLayers, _extended,
                                                                                        _keypointsRatio, _upright);
    referenceFeatureDetector = std::make_shared<providentia::features::SurfFeatureDetector>(_hessianThreshold,
                                                                                            _nOctaves, _nOctaveLayers,
                                                                                            _extended, _keypointsRatio,
                                                                                            _upright);
    matcher = std::make_shared<providentia::features::BruteForceFeatureMatcher>(cv::NORM_L2);
}

#pragma endregion SURFBFDynamicStabilizer
