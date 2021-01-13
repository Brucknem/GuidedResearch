//
// Created by brucknem on 12.01.21.
//
#include <opencv2/cudaimgproc.hpp>
#include <utility>
#include "FeatureDetection.hpp"

#pragma region Getters_Setters{

void providentia::features::FeatureDetectorBase::setCurrentMask(cv::Size _size) {
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

void providentia::features::FeatureDetectorBase::detect(const cv::cuda::GpuMat &_frame) {
    clear();
    originalFrame = _frame.clone();
    cv::cuda::cvtColor(_frame, frame, cv::COLOR_BGR2GRAY);
    specificDetect();
    addTimestamp("Detection finished", 0);
}

void providentia::features::FeatureDetectorBase::detect(const cv::cuda::GpuMat &_frame, bool _useLatestMask) {
    useLatestMask = _useLatestMask;
    detect(_frame);
}

void providentia::features::FeatureDetectorBase::detect(const cv::cuda::GpuMat &_frame, const cv::cuda::GpuMat &_mask) {
    latestMask = _mask;
    detect(_frame, true);
}

cv::Mat providentia::features::FeatureDetectorBase::draw() {
    cv::drawKeypoints(cv::Mat(originalFrame), getKeypointsCPU(), drawFrame);
    return drawFrame;
}

const cv::cuda::GpuMat &providentia::features::FeatureDetectorBase::getOriginalFrame() const {
    return originalFrame;
}

const cv::cuda::GpuMat &providentia::features::FeatureDetectorBase::getDescriptorsGPU() const {
    return descriptorsGPU;
}

bool providentia::features::FeatureDetectorBase::isEmpty() {
    return keypointsGPU.empty();
}

providentia::features::FeatureDetectorBase::FeatureDetectorBase() : providentia::utils::TimeMeasurable(
        "FeatureDetectorBase", 1) {}

const cv::cuda::GpuMat &providentia::features::FeatureDetectorBase::getCurrentMask(cv::Size _size) {
    setCurrentMask(std::move(_size));
    return currentMask;
}

#pragma endregion FeatureDetectorBase

#pragma region SurfFeatureDetector

providentia::features::SurfFeatureDetector::SurfFeatureDetector(double _hessianThreshold, int _nOctaves,
                                                                int _nOctaveLayers, bool _extended,
                                                                float _keypointsRatio, bool _upright) {
    detector = cv::cuda::SURF_CUDA::create(_hessianThreshold, _nOctaves, _nOctaveLayers, _extended,
                                           _keypointsRatio, _upright);
    setName(typeid(*this).name());
}

void providentia::features::SurfFeatureDetector::specificDetect() {
    setCurrentMask();
    detector->detectWithDescriptors(frame, currentMask, keypointsGPU, descriptorsGPU, false);
}

const std::vector<cv::KeyPoint> &providentia::features::SurfFeatureDetector::getKeypointsCPU() {
    detector->downloadKeypoints(keypointsGPU, keypointsCPU);
    return keypointsCPU;
}


#pragma endregion SurfFeatureDetector