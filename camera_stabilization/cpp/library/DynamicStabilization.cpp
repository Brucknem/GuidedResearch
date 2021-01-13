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
        referenceFeatureDetector->detect(_frame, segmentor->getBackgroundMask(_frame.size()).clone());
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
    frameFeatureDetector = std::make_shared<providentia::features::SurfFeatureDetector>(_hessianThreshold, _nOctaves,
                                                                                        _nOctaveLayers, _extended,
                                                                                        _keypointsRatio, _upright);
    referenceFeatureDetector = std::make_shared<providentia::features::SurfFeatureDetector>(_hessianThreshold,
                                                                                            _nOctaves, _nOctaveLayers,
                                                                                            _extended, _keypointsRatio,
                                                                                            _upright);
    matcher = std::make_shared<providentia::features::BruteForceFeatureMatcher>(cv::NORM_L2);
    setName(typeid(*this).name());
}

#pragma endregion SURFBFDynamicStabilizer
