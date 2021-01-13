//
// Created by brucknem on 13.01.21.
//

#include "BackgroundSegmentation.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudabgsegm.hpp"

#pragma region BackgroundSegmentorBase

void providentia::segmentation::BackgroundSegmentorBase::apply(const cv::cuda::GpuMat &_frame) {
    specificApply(_frame);
    if (all255Mask.empty() || all255Mask.size() != foregroundMask.size()) {
        all255Mask.upload(cv::Mat::ones(foregroundMask.size(), CV_8UC1) * 255);
    }

    postProcess();
    cv::cuda::absdiff(all255Mask, foregroundMask, backgroundMask, stream);
    stream.waitForCompletion();
}

void providentia::segmentation::BackgroundSegmentorBase::postProcess() {

}

const cv::cuda::GpuMat &providentia::segmentation::BackgroundSegmentorBase::getForegroundMask() const {
    return foregroundMask;
}

const cv::cuda::GpuMat &providentia::segmentation::BackgroundSegmentorBase::getBackgroundMask() const {
    return backgroundMask;
}

#pragma endregion BackgroundSegmentorBase

#pragma region MOG2BackgroundSegmentor

providentia::segmentation::MOG2BackgroundSegmentor::MOG2BackgroundSegmentor(int history, double varThreshold,
                                                                            bool detectShadows) {
    algorithm = cv::cuda::createBackgroundSubtractorMOG2(history, varThreshold, detectShadows);
}

void providentia::segmentation::MOG2BackgroundSegmentor::specificApply(const cv::cuda::GpuMat &_frame) {
    algorithm->apply(_frame, foregroundMask, -1, stream);
}

#pragma endregion MOG2BackgroundSegmentor
