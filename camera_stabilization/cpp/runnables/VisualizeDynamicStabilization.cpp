//
// Created by brucknem on 13.01.21.
//
#include <opencv2/cudawarping.hpp>
#include "DynamicStabilization.hpp"
#include "Commons.hpp"

/**
 * Setup to visualize the total stabilization algorithm.
 */
class Setup : public providentia::runnables::BaseSetup {
private:
    /**
     * The matcher used to match the features.
     */
    std::shared_ptr<providentia::stabilization::DynamicStabilizerBase> stabilizer;
    cv::Mat referenceMask, currentMask;
public:
    explicit Setup(int argc, char const *argv[]) : BaseSetup(argc, argv) {
//        stabilizer = std::make_shared<providentia::stabilization::SURFBFDynamicStabilizer>();
//        stabilizer = std::make_shared<providentia::stabilization::ORBBFDynamicStabilizer>();
        stabilizer = std::make_shared<providentia::stabilization::FastFREAKBFDynamicStabilizer>();
    }

    void specificMainLoop() override {
        stabilizer->stabilize(frameGPU);
        totalAlgorithmsDuration = stabilizer->getTotalMilliseconds();
        cv::hconcat(std::vector<cv::Mat>{stabilizer->draw(), cv::Mat(stabilizer->getReferenceframe())}, finalFrame);
        cv::cvtColor(cv::Mat(stabilizer->getSegmentor()->getBackgroundMask(frameGPU.size())), currentMask,
                     cv::COLOR_GRAY2BGR);
        cv::cvtColor(cv::Mat(stabilizer->getReferenceframeMask()), referenceMask, cv::COLOR_GRAY2BGR);
        cv::hconcat(
                std::vector<cv::Mat>{cv::Mat::zeros(frameGPU.size(), CV_8UC3), currentMask, referenceMask},
                bufferCPU);
        cv::vconcat(std::vector<cv::Mat>{finalFrame, bufferCPU}, finalFrame);

//        cv::imshow("Test", stabilizer->getMatcher()->draw());
    }

    void specificAddMessages() override {
        addRuntimeToFinalFrame("Feature detection", stabilizer->getFrameFeatureDetector()->getTotalMilliseconds(),
                               5, 20);
        addRuntimeToFinalFrame("Feature matching",
                               stabilizer->getMatcher()->getTotalMilliseconds(), 5, 40);
        addRuntimeToFinalFrame("Frame warping ",
                               stabilizer->getWarper()->getTotalMilliseconds(), 5, 60);
        addRuntimeToFinalFrame("Background segmentation",
                               stabilizer->getSegmentor()->getTotalMilliseconds(), 5, 80);
        addRuntimeToFinalFrame("Total stabilization", stabilizer->getTotalMilliseconds(), 5, 100);
    }
};

int main(int argc, char const *argv[]) {
    Setup setup(argc, argv);
    setup.mainLoop();
    return 0;
}
