//
// Created by brucknem on 13.01.21.
//
#include <opencv2/cudawarping.hpp>
#include "DynamicStabilization.hpp"
#include "Commons.hpp"
#include "OpticalFlow.hpp"

/**
 * Setup to visualize the optical flow.
 */
class Setup : public providentia::runnables::BaseSetup {
private:
    /**
     * The stabilizer used to stabilize the video.
     */
    std::shared_ptr<providentia::stabilization::DynamicStabilizerBase> stabilizer;

    /**
     * The optical flow algorithms for the original and stabilized frames.
     */
    std::shared_ptr<providentia::opticalflow::DenseOpticalFlow> stabilizedOpticalFlow, originalOpticalFlow;

public:
    explicit Setup(int argc, char const *argv[]) : BaseSetup(argc, argv) {
        stabilizer = std::make_shared<providentia::stabilization::SURFBFDynamicStabilizer>();

        originalOpticalFlow = std::make_shared<providentia::opticalflow::FarnebackDenseOpticalFlow>();
        stabilizedOpticalFlow = std::make_shared<providentia::opticalflow::FarnebackDenseOpticalFlow>();
    }

    void specificMainLoop() override {
        stabilizer->stabilize(frameGPU);
        originalOpticalFlow->calculate(frameGPU);
        stabilizedOpticalFlow->calculate(stabilizer->getStabilizedFrame());

        totalAlgorithmsDuration = stabilizer->getTotalMilliseconds() + originalOpticalFlow->getTotalMilliseconds() +
                                  stabilizedOpticalFlow->getTotalMilliseconds();


        cv::hconcat(std::vector<cv::Mat>{frameCPU, cv::Mat(stabilizer->getStabilizedFrame())}, finalFrame);

        cv::hconcat(std::vector<cv::Mat>{originalOpticalFlow->draw(), stabilizedOpticalFlow->draw()}, bufferCPU);
        cv::vconcat(std::vector<cv::Mat>{finalFrame, bufferCPU}, finalFrame);
    }

    void specificAddMessages() override {
        addRuntimeToFinalFrame("Stabilization", stabilizer->getTotalMilliseconds(), 5, 20);
        addRuntimeToFinalFrame("Optical Flow [Original]", originalOpticalFlow->getTotalMilliseconds(), 5, 40);
        addRuntimeToFinalFrame("Optical Flow [Stabilized]", stabilizedOpticalFlow->getTotalMilliseconds(), 5, 60);
    }
};

int main(int argc, char const *argv[]) {
    Setup setup(argc, argv);
    setup.mainLoop();
    return 0;
}
