//
// Created by brucknem on 13.01.21.
//
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudastereo.hpp>
#include "opencv2/cudaimgproc.hpp"
#include "Commons.hpp"


/**
 * Setup to visualize the optical flow.
 */
class Setup : public providentia::runnables::BaseSetup {
private:
    cv::Ptr<cv::cuda::StereoBM> stereo;
//    cv::Ptr<cv::cuda::StereoSGM> stereo;

    cv::cuda::GpuMat right;

public:
    explicit Setup(int argc, char const *argv[]) : BaseSetup(argc, argv) {
        stereo = cv::cuda::createStereoBM(4, 15);
//        stereo = cv::cuda::createStereoSGM();
        right.upload(cv::imread("../test/s40_n_cam_near_calibration_test_image.png"));
        cv::cuda::cvtColor(right, right, cv::COLOR_BGR2GRAY);
    }

    void specificMainLoop() override {
        cv::cuda::cvtColor(frameGPU, frameGPU, cv::COLOR_BGR2GRAY);
        stereo->compute(frameGPU, right, bufferGPU);
        bufferGPU.download(finalFrame);
    }

    void specificAddMessages() override {
    }
};

int main(int argc, char const *argv[]) {
    Setup setup(argc, argv);
    setup.mainLoop();
    return 0;
}
