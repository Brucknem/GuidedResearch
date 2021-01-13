//
// Created by brucknem on 13.01.21.
//
#include <opencv2/cudawarping.hpp>
#include "FeatureDetection.hpp"
#include "Commons.hpp"

/**
 * Setup to visualize the feature detection.
 */
class Setup : public providentia::runnables::BaseSetup {
private:
    /**
     * The detectors and scaling factors that are applied in the main loop.
     */
    std::vector<std::pair<providentia::features::FeatureDetectorBase *, float>> detectors;

public:
    explicit Setup(int argc, char const *argv[]) : BaseSetup(argc, argv) {
        detectors.emplace_back(std::make_pair(new providentia::features::SURFFeatureDetector(), 1.0));
//        detectors.emplace_back(std::make_pair(new providentia::features::SURFFeatureDetector(), 0.75));
//        detectors.emplace_back(std::make_pair(new providentia::features::SURFFeatureDetector(), 0.5));
        detectors.emplace_back(std::make_pair(new providentia::features::ORBFeatureDetector(), 1.0));
//        detectors.emplace_back(std::make_pair(new providentia::features::ORBFeatureDetector(), 0.75));
//        detectors.emplace_back(std::make_pair(new providentia::features::ORBFeatureDetector(), 0.5));
        detectors.emplace_back(std::make_pair(new providentia::features::FastFeatureDetector(), 1.0));
//        detectors.emplace_back(std::make_pair(new providentia::features::FastFeatureDetector(), 0.75));
//        detectors.emplace_back(std::make_pair(new providentia::features::FastFeatureDetector(), 0.5));
    }

    void specificMainLoop() override {
        for (const auto &entry : detectors) {
            cv::cuda::resize(frameGPU, bufferGPU, cv::Size(), entry.second, entry.second);
            entry.first->detect(bufferGPU);
            totalAlgorithmsDuration += entry.first->getTotalMilliseconds();
            bufferCPU = entry.first->draw();
            cv::resize(bufferCPU, bufferCPU, frameCPU.size());
            providentia::runnables::addRuntimeToFrame(bufferCPU,
                                                      std::string(typeid(*entry.first).name()) + " [" +
                                                      std::to_string(entry.second) + "]",
                                                      entry.first->getTotalMilliseconds(),
                                                      5, 20);

            if (finalFrame.empty()) {
                finalFrame = bufferCPU.clone();
            } else {
                cv::hconcat(std::vector<cv::Mat>{finalFrame, bufferCPU}, finalFrame);
            }
        }
    }
};

int main(int argc, char const *argv[]) {
    Setup setup(argc, argv);
    setup.setRenderingScaleFactor(1);
    setup.mainLoop();
    return 0;
}
