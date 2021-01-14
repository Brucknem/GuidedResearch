#include "opencv2/imgproc/imgproc.hpp"

#include "DynamicStabilization.hpp"
#include <opencv2/cudaarithm.hpp>

#pragma region DynamicStabilizerBase

void providentia::stabilization::DynamicStabilizerBase::stabilize(const cv::cuda::GpuMat &_frame) {
    clear();
    updateKeyframe();

    frameFeatureDetector->detect(_frame, segmentor->getBackgroundMask(_frame.size()));
    if (referenceFeatureDetector->isEmpty()) {
        // TODO copy/clone frameFeatureDetector
        referenceFeatureDetector->detect(_frame, segmentor->getBackgroundMask(_frame.size()));
    }

    matcher->match(frameFeatureDetector, referenceFeatureDetector);
    warper->warp(_frame, matcher);
    segmentor->segment(getStabilizedFrame());
    addTimestamp("Stabilization finished", 0);
}

void providentia::stabilization::DynamicStabilizerBase::updateKeyframe() {
    if (currentIteration++ < warmUp) {
        return;
    }

    int newCount = cv::cuda::countNonZero(this->segmentor->getBackgroundMask(getStabilizedFrame().size()));
    int oldCount = cv::cuda::countNonZero(this->referenceFeatureDetector->getCurrentMask());
    if (oldCount == this->getStabilizedFrame().size().width * this->getStabilizedFrame().size().height ||
        newCount > oldCount) {
        this->referenceFeatureDetector->detect(this->getStabilizedFrame(),
                                               this->segmentor->getBackgroundMask(getStabilizedFrame().size()).clone());
    }
}

const cv::Mat &providentia::stabilization::DynamicStabilizerBase::getHomography() const {
    return warper->getHomography();
}

const cv::cuda::GpuMat &providentia::stabilization::DynamicStabilizerBase::getStabilizedFrame() const {
    return warper->getWarpedFrame();
}

providentia::stabilization::DynamicStabilizerBase::DynamicStabilizerBase() : providentia::utils::TimeMeasurable(
        "DynamicStabilizerBase", 1) {
    warper = std::make_shared<providentia::stabilization::FrameWarper>();
    segmentor = std::make_shared<providentia::segmentation::MOG2BackgroundSegmentor>(cv::Size(1920, 1200) / 10);
//    segmentor = std::make_shared<providentia::segmentation::MockBackgroundSegmentor>();
}

cv::Mat providentia::stabilization::DynamicStabilizerBase::draw() {
    cv::Mat result;
    cv::hconcat(std::vector<cv::Mat>{cv::Mat(frameFeatureDetector->getOriginalFrame()), cv::Mat(getStabilizedFrame())},
                result);
    return result;
}

const cv::cuda::GpuMat &providentia::stabilization::DynamicStabilizerBase::getReferenceframe() const {
    return referenceFeatureDetector->getOriginalFrame();
}

const cv::cuda::GpuMat &providentia::stabilization::DynamicStabilizerBase::getReferenceframeMask() const {
    return referenceFeatureDetector->getCurrentMask();
}

const std::shared_ptr<providentia::features::FeatureDetectorBase> &
providentia::stabilization::DynamicStabilizerBase::getFrameFeatureDetector() const {
    return frameFeatureDetector;
}

const std::shared_ptr<providentia::features::FeatureMatcherBase> &
providentia::stabilization::DynamicStabilizerBase::getMatcher() const {
    return matcher;
}

const std::shared_ptr<providentia::segmentation::BackgroundSegmentorBase> &
providentia::stabilization::DynamicStabilizerBase::getSegmentor() const {
    return segmentor;
}

const std::shared_ptr<providentia::stabilization::FrameWarper> &
providentia::stabilization::DynamicStabilizerBase::getWarper() const {
    return warper;
}

#pragma endregion DynamicStabilizerBase

#pragma region SURFBFDynamicStabilizer

providentia::stabilization::SURFBFDynamicStabilizer::SURFBFDynamicStabilizer(double _hessianThreshold,
                                                                             int _nOctaves,
                                                                             int _nOctaveLayers,
                                                                             bool _extended,
                                                                             float _keypointsRatio,
                                                                             bool _upright) {
    auto detector = providentia::features::SURFFeatureDetector(_hessianThreshold, _nOctaves,
                                                               _nOctaveLayers, _extended,
                                                               _keypointsRatio, _upright);
    frameFeatureDetector = std::make_shared<providentia::features::SURFFeatureDetector>(detector);
    referenceFeatureDetector = std::make_shared<providentia::features::SURFFeatureDetector>(detector);
    matcher = std::make_shared<providentia::features::BruteForceFeatureMatcher>(cv::NORM_L2);
    setName(typeid(*this).name());
}

#pragma endregion SURFBFDynamicStabilizer

#pragma region ORBBFDynamicStabilizer

providentia::stabilization::ORBBFDynamicStabilizer::ORBBFDynamicStabilizer(int nfeatures, float scaleFactor,
                                                                           int nlevels, int edgeThreshold,
                                                                           int firstLevel, int WTA_K, int scoreType,
                                                                           int patchSize, int fastThreshold,
                                                                           bool blurForDescriptor) {
    auto detector = providentia::features::ORBFeatureDetector(nfeatures, scaleFactor,
                                                              nlevels, edgeThreshold,
                                                              firstLevel, WTA_K, scoreType,
                                                              patchSize, fastThreshold,
                                                              blurForDescriptor);
    frameFeatureDetector = std::make_shared<providentia::features::ORBFeatureDetector>(detector);
    referenceFeatureDetector = std::make_shared<providentia::features::ORBFeatureDetector>(detector);
    matcher = std::make_shared<providentia::features::BruteForceFeatureMatcher>(cv::NORM_HAMMING);
    setName(typeid(*this).name());
}

#pragma endregion ORBBFDynamicStabilizer

providentia::stabilization::FastFREAKBFDynamicStabilizer::FastFREAKBFDynamicStabilizer(int threshold,
                                                                                       bool nonmaxSuppression,
                                                                                       cv::FastFeatureDetector::DetectorType type,
                                                                                       int max_npoints,
                                                                                       bool orientationNormalized,
                                                                                       bool scaleNormalized,
                                                                                       float patternScale,
                                                                                       int nOctaves,
                                                                                       const std::vector<int> &selectedPairs) {
    auto detector = providentia::features::FastFREAKFeatureDetector(threshold,
                                                                    nonmaxSuppression, type,
                                                                    max_npoints,
                                                                    orientationNormalized,
                                                                    scaleNormalized,
                                                                    patternScale, nOctaves,
                                                                    selectedPairs);
    frameFeatureDetector = std::make_shared<providentia::features::FastFREAKFeatureDetector>(detector);
    referenceFeatureDetector = std::make_shared<providentia::features::FastFREAKFeatureDetector>(detector);
    matcher = std::make_shared<providentia::features::BruteForceFeatureMatcher>(cv::NORM_HAMMING);
    setName(typeid(*this).name());
}
