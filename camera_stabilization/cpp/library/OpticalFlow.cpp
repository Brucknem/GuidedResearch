//
// Created by brucknem on 18.01.21.
//

#include "OpticalFlow.hpp"

#pragma region DenseOpticalFlow

void providentia::opticalflow::DenseOpticalFlow::initialize() {
    previousFrame = currentFrame.clone();
    hsv = cv::Mat(currentFrame.size(), CV_8UC3, cv::Scalar(255));
    bgr = cv::Mat(currentFrame.size(), CV_8UC3, cv::Scalar(255));
}

void providentia::opticalflow::DenseOpticalFlow::calculate(const cv::cuda::GpuMat &_frame) {
    clear();

    cv::cuda::cvtColor(_frame, currentFrame, cv::COLOR_BGR2GRAY);

    if (previousFrame.empty()) {
        initialize();
        return;
    }

    specificCalculate();

    denseOpticalFlowGPU.download(denseOpticalFlowCPU);

    cv::split(denseOpticalFlowCPU, flowParts);

    cv::cartToPolar(flowParts[0], flowParts[1], flowParts[0], flowParts[1], true);
    magnitude = flowParts[0];
    angle = flowParts[1];
    normalize(flowParts[0], flowParts[2], 0.0f, 1.0f, cv::NORM_MINMAX);
    flowParts[1] *= ((1.f / 360.f) * (180.f / 255.f));

    //build hsv image
    _hsv[0] = flowParts[1];
    _hsv[1] = cv::Mat::ones(flowParts[1].size(), CV_32F);
    _hsv[2] = flowParts[2];
    merge(_hsv, 3, hsv);

    hsv.convertTo(hsv8, CV_8U, 255.0);
    cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

    previousFrame = currentFrame.clone();
    addTimestamp("Optical Flow calculation finished, 0");
}

double providentia::opticalflow::DenseOpticalFlow::getMagnitudeMean() {
    return cv::mean(magnitude)[0];
}

double providentia::opticalflow::DenseOpticalFlow::getAngleMean() {
    return cv::mean(angle)[0];
}

const cv::Mat &providentia::opticalflow::DenseOpticalFlow::draw() const {
    return bgr;
}

providentia::opticalflow::DenseOpticalFlow::~DenseOpticalFlow() = default;

#pragma endregion DenseOpticalFlow

#pragma region FarnebackDenseOpticalFlow

void providentia::opticalflow::FarnebackDenseOpticalFlow::specificCalculate() {
    opticalFlow->calc(previousFrame, currentFrame, denseOpticalFlowGPU, stream);
}

providentia::opticalflow::FarnebackDenseOpticalFlow::FarnebackDenseOpticalFlow() : DenseOpticalFlow() {
    opticalFlow = cv::cuda::FarnebackOpticalFlow::create();
    setName(typeid(*this).name());
}

providentia::opticalflow::FarnebackDenseOpticalFlow::~FarnebackDenseOpticalFlow() = default;

#pragma endregion FarnebackDenseOpticalFlow
