//
// Created by brucknem on 13.01.21.
//
#include "BackgroundSegmentation.hpp"
#include "Commons.hpp"

/**
 * Setup to visualize the background segmentation.
 */
class Setup : public providentia::runnable::BaseSetup {
private:
	/**
	 * The matcher used to match the features.
	 */
	std::shared_ptr<providentia::segmentation::BackgroundSegmentorBase> segmentor;

public:
	explicit Setup(int argc, char const *argv[]) : BaseSetup(argc, argv) {
		segmentor = std::make_shared<providentia::segmentation::MOG2BackgroundSegmentor>();
	}

	void specificMainLoop() override {
		segmentor->segment(frameGPU);
		totalAlgorithmsDuration = segmentor->getTotalMilliseconds();
		finalFrame = segmentor->draw();
	}

	void specificAddMessages() override {
		addRuntimeToFinalFrame("MOG2 Background segmentation", segmentor->getTotalMilliseconds(), 5, 20);
	}
};

int main(int argc, char const *argv[]) {
	Setup setup(argc, argv);
	setup.mainLoop();
	return 0;
}
