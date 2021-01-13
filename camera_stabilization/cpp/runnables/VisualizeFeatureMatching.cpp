//
// Created by brucknem on 13.01.21.
//
#include "FeatureDetection.hpp"
#include "FeatureMatching.hpp"
#include "RunnablesCommons.hpp"

/**
 * Setup to visualize the feature matching.
 */
class Setup : public providentia::runnables::BaseSetup {
private:
    /**
     * The current frame feature detector applied in the main loop.
     */
    std::shared_ptr<providentia::features::SurfFeatureDetector> frameDetector;

    /**
     * The reference frame feature detector applied in the main loop.
     */
    std::shared_ptr<providentia::features::SurfFeatureDetector> referenceFrameDetector;

    /**
     * The matcher used to match the features.
     */
    std::shared_ptr<providentia::features::BruteForceFeatureMatcher> matcher;

public:
    explicit Setup(int argc, char const *argv[]) : BaseSetup(argc, argv) {
        frameDetector = std::make_shared<providentia::features::SurfFeatureDetector>();
        referenceFrameDetector = std::make_shared<providentia::features::SurfFeatureDetector>();
        matcher = std::make_shared<providentia::features::BruteForceFeatureMatcher>(cv::NORM_L2);
    }

    void specificMainLoop() override {
        frameDetector->detect(frameGPU);
        if (referenceFrameDetector->isEmpty()) {
            referenceFrameDetector->detect(frameGPU);
        }

        matcher->match(frameDetector, referenceFrameDetector);
        totalAlgorithmsDuration = frameDetector->getTotalMilliseconds() + matcher->getTotalMilliseconds();

        finalFrame = matcher->draw();
    }

    void specificAddMessages() override {
        addRuntimeToFinalFrame("SURF detection", frameDetector->getTotalMilliseconds(), 5, 20);
        addRuntimeToFinalFrame("Brute Force Matching", matcher->getTotalMilliseconds(), 5, 40);
    }
};

int main(int argc, char const *argv[]) {
    Setup setup(argc, argv);
    setup.mainLoop();
    return 0;
}
