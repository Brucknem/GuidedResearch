//
// Created by brucknem on 13.01.21.
//
#include "Commons.hpp"

/**
 * Setup to visualize the background segmentation.
 */
class Setup : public providentia::evaluation::VideoSetup {
private:
	/**
	 * The matcher used to match the features.
	 */
	std::shared_ptr<providentia::segmentation::BackgroundSegmentionBase> segmentor;

public:
	explicit Setup() : VideoSetup() {
		segmentor = std::make_shared<providentia::segmentation::MOG2BackgroundSegmention>();
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
	Setup setup;
	setup.mainLoop();
	return 0;
}
