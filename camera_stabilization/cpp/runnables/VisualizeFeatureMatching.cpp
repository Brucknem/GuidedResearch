//
// Created by brucknem on 13.01.21.
//
#include "FeatureDetection.hpp"
#include "FeatureMatching.hpp"
#include "Commons.hpp"

/**
 * Setup to visualize the feature matching.
 */
class Setup : public providentia::runnables::BaseSetup {
private:
    /**
     * The current frame feature detector applied in the main loop.
     */
    std::shared_ptr<providentia::features::FeatureDetectorBase> frameDetector, referenceFrameDetector;

    /**
     * The matcher used to match the features.
     */
    std::shared_ptr<providentia::features::FeatureMatcherBase> matcher, matcherWithoutFundamental;

public:
    explicit Setup(int argc, char const *argv[]) : BaseSetup(argc, argv) {
//        providentia::features::SIFTFeatureDetector detector(100);
//        frameDetector = std::make_shared<providentia::features::SIFTFeatureDetector>(detector);
//        referenceFrameDetector = std::make_shared<providentia::features::SIFTFeatureDetector>(detector);

        providentia::features::SURFFeatureDetector detector;
        frameDetector = std::make_shared<providentia::features::SURFFeatureDetector>(detector);
        referenceFrameDetector = std::make_shared<providentia::features::SURFFeatureDetector>(detector);

//        providentia::features::ORBFeatureDetector detector(1000);
//        frameDetector = std::make_shared<providentia::features::ORBFeatureDetector>(detector);
//        referenceFrameDetector = std::make_shared<providentia::features::ORBFeatureDetector>(detector);

//        providentia::features::FastFREAKFeatureDetector detector;
//        frameDetector = std::make_shared<providentia::features::FastFREAKFeatureDetector>(detector);
//        referenceFrameDetector = std::make_shared<providentia::features::FastFREAKFeatureDetector>(detector);

        matcher = std::make_shared<providentia::features::BruteForceFeatureMatcher>(cv::NORM_L2);
        matcher->setShouldUseFundamentalMatrix(false);
        matcherWithoutFundamental = std::make_shared<providentia::features::BruteForceFeatureMatcher>(cv::NORM_L2);
//        matcher = std::make_shared<providentia::features::BruteForceFeatureMatcher>(cv::NORM_HAMMING);
//        matcher = std::make_shared<providentia::features::FlannFeatureMatcher>(true);
    }

    void specificMainLoop() override {
        frameDetector->detect(frameGPU);
        if (referenceFrameDetector->isEmpty()) {
            referenceFrameDetector->detect(frameGPU);
        }

        matcher->match(frameDetector, referenceFrameDetector);
        matcherWithoutFundamental->match(frameDetector, referenceFrameDetector);
        totalAlgorithmsDuration = frameDetector->getTotalMilliseconds() + matcher->getTotalMilliseconds() +
                                  matcherWithoutFundamental->getTotalMilliseconds();

        cv::vconcat(matcherWithoutFundamental->draw(), matcher->draw(), finalFrame);
    }

    void specificAddMessages() override {
        addRuntimeToFinalFrame("Feature detection", frameDetector->getTotalMilliseconds(), 5, 20);
        addRuntimeToFinalFrame("Feature matching", matcher->getTotalMilliseconds(), 5, 40);
    }
};

int main(int argc, char const *argv[]) {
    Setup setup(argc, argv);
    setup.mainLoop();
    return 0;
}
