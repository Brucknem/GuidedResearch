//
// Created by brucknem on 12.01.21.
//
#include <opencv2/cudaimgproc.hpp>
#include "opencv2/features2d.hpp"
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
    setCurrentMask();
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

const cv::cuda::GpuMat &providentia::features::FeatureDetectorBase::getOriginalFrame() const {
    return originalFrame;
}

providentia::features::FeatureDetectorBase::FeatureDetectorBase() : providentia::utils::TimeMeasurable(
        "FeatureDetectorBase", 1) {}

const cv::cuda::GpuMat &providentia::features::FeatureDetectorBase::getCurrentMask(cv::Size _size) {
    setCurrentMask(std::move(_size));
    return currentMask;
}

const std::vector<cv::KeyPoint> &providentia::features::FeatureDetectorBase::getKeypoints() const {
    return keypointsCPU;
}

bool providentia::features::FeatureDetectorBase::isEmpty() {
    return keypointsCPU.empty();
}

cv::Mat providentia::features::FeatureDetectorBase::draw() {
    cv::drawKeypoints(cv::Mat(originalFrame), keypointsCPU, drawFrame);
    return drawFrame;
}

const cv::cuda::GpuMat &providentia::features::FeatureDetectorBase::getDescriptorsGPU() const {
    return descriptorsGPU;
}

const cv::Mat &providentia::features::FeatureDetectorBase::getDescriptorsCPU() const {
    return descriptorsCPU;
}

#pragma endregion FeatureDetectorBase

#pragma region SURFFeatureDetector

providentia::features::SURFFeatureDetector::SURFFeatureDetector(double _hessianThreshold, int _nOctaves,
                                                                int _nOctaveLayers, bool _extended,
                                                                float _keypointsRatio, bool _upright) {
    detector = cv::cuda::SURF_CUDA::create(_hessianThreshold, _nOctaves, _nOctaveLayers, _extended,
                                           _keypointsRatio, _upright);
    setName(typeid(*this).name());
}

void providentia::features::SURFFeatureDetector::specificDetect() {
    detector->detectWithDescriptors(frame, getCurrentMask(frame.size()), keypointsGPU, descriptorsGPU, false);
    detector->downloadKeypoints(keypointsGPU, keypointsCPU);
    descriptorsGPU.download(descriptorsCPU);
}

#pragma endregion SURFFeatureDetector

#pragma region ORBFeatureDetector

providentia::features::ORBFeatureDetector::ORBFeatureDetector(int nfeatures, float scaleFactor, int nlevels,
                                                              int edgeThreshold, int firstLevel, int WTA_K,
                                                              int scoreType, int patchSize, int fastThreshold,
                                                              bool blurForDescriptor) {
    detector = cv::cuda::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType,
                                     patchSize, fastThreshold, blurForDescriptor);
    setName(typeid(*this).name());
}

void providentia::features::ORBFeatureDetector::specificDetect() {
    detector->detectAndComputeAsync(frame, getCurrentMask(frame.size()), keypointsGPU, descriptorsGPU, false);
    detector->convert(keypointsGPU, keypointsCPU);
    descriptorsGPU.download(descriptorsCPU);
}

#pragma endregion ORBFeatureDetector

#pragma region FastFeatureDetector

providentia::features::FastFREAKFeatureDetector::FastFREAKFeatureDetector(int threshold, bool nonmaxSuppression,
                                                                          cv::FastFeatureDetector::DetectorType type,
                                                                          int max_npoints, bool orientationNormalized,
                                                                          bool scaleNormalized, float patternScale,
                                                                          int nOctaves,
                                                                          const std::vector<int> &selectedPairs) {
    detector = cv::FastFeatureDetector::create(threshold, nonmaxSuppression, type);
    descriptor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves);
    setName(typeid(*this).name());
}

void providentia::features::FastFREAKFeatureDetector::specificDetect() {
    detector->detect(cv::Mat(frame), keypointsCPU, cv::Mat(getCurrentMask(frame.size())));
    descriptor->compute(cv::Mat(frame), keypointsCPU, descriptorsCPU);
}

#pragma endregion FastFeatureDetector

providentia::features::SIFTFeatureDetector::SIFTFeatureDetector(int nfeatures, int nOctaveLayers,
                                                                double contrastThreshold, double edgeThreshold,
                                                                double sigma) {
    detector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    setName(typeid(*this).name());
}

void providentia::features::SIFTFeatureDetector::specificDetect() {
    detector->detectAndCompute(cv::Mat(frame), cv::Mat(getCurrentMask()), keypointsCPU, descriptorsCPU);
    descriptorsGPU.upload(descriptorsCPU);
}

