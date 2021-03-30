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
class Setup : public providentia::evaluation::VideoSetup {
private:
	/**
	 * The stabilizer used to stabilize the video.
	 */
	std::shared_ptr<providentia::stabilization::DynamicStabilizerBase> stabilizer;

	/**
	 * The optical flow algorithms for the original and stabilized writeFrames.
	 */
	std::shared_ptr<providentia::opticalflow::DenseOpticalFlow> stabilizedOpticalFlow, originalOpticalFlow;

public:
	explicit Setup() : VideoSetup() {
		stabilizer = std::make_shared<providentia::stabilization::SURFBFDynamicStabilizer>();
		stabilizer->setShouldUpdateKeyframe(true);

		originalOpticalFlow = std::make_shared<providentia::opticalflow::FarnebackDenseOpticalFlow>();
		stabilizedOpticalFlow = std::make_shared<providentia::opticalflow::FarnebackDenseOpticalFlow>();
	}

	void specificMainLoop() override {
		stabilizer->stabilize(frameGPU);
		originalOpticalFlow->calculate(frameGPU);
		stabilizedOpticalFlow->calculate(stabilizer->getStabilizedFrame());

//        totalAlgorithmsDuration = stabilizer->getTotalMilliseconds() + originalOpticalFlow->getTotalMilliseconds() +
//                                  stabilizedOpticalFlow->getTotalMilliseconds();

//        cv::imshow("Keyframe", cv::Mat(stabilizer->getReferenceframe()));

		cv::hconcat(std::vector<cv::Mat>{frameCPU, cv::Mat(stabilizer->getStabilizedFrame())}, finalFrame);

		cv::hconcat(std::vector<cv::Mat>{originalOpticalFlow->draw(), stabilizedOpticalFlow->draw()}, bufferCPU);
		cv::vconcat(std::vector<cv::Mat>{finalFrame, bufferCPU}, finalFrame);
	}

	void specificAddMessages() override {
		addRuntimeToFinalFrame("Original", 1, 5, 20);
		addRuntimeToFinalFrame("Stabilized", stabilizer->getTotalMilliseconds(), finalFrame.cols / 2 + 5, 20);

//        addRuntimeToFinalFrame("Optical Flow [Original]", originalOpticalFlow->getTotalMilliseconds(), 5,
//                               finalFrame.rows / 2 + 20);
		addTextToFinalFrame(
			"Optical Flow [Original] - Mean expectedPixel shift - " +
			std::to_string(originalOpticalFlow->getMagnitudeMean()) + " px", 5,
			finalFrame.rows / 2 + 20);


//        addRuntimeToFinalFrame("Optical Flow [Stabilized]", stabilizedOpticalFlow->getTotalMilliseconds(),
//                               finalFrame.cols / 2 + 5, finalFrame.rows / 2 + 20);
		addTextToFinalFrame(
			"Optical Flow [Stabilized] - Mean expectedPixel shift - " +
			std::to_string(stabilizedOpticalFlow->getMagnitudeMean()) + " px", finalFrame.cols / 2 + 5,
			finalFrame.rows / 2 + 20);
	}
};

int main(int argc, char const *argv[]) {
	Setup setup;
	setup.mainLoop();
	return 0;
}
