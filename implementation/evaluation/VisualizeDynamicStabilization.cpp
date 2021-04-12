//
// Created by brucknem on 13.01.21.
//
#include <opencv2/cudawarping.hpp>

#include "DynamicStabilizationBase.hpp"
#include "SURFBFDynamicStabilization.hpp"
#include "ORBBFDynamicStabilization.hpp"
#include "FastFREAKBFDynamicStabilization.hpp"

#include "OpticalFlow.hpp"
#include "Commons.hpp"

using namespace providentia::evaluation;

/**
 * Setup to visualize the total stabilization algorithm.
 */
class Setup : public ::VideoSetup {
private:
	/**
	 * The matcher used to match the features.
	 */
	std::shared_ptr<providentia::stabilization::DynamicStabilizationBase> stabilizer;
	std::vector<providentia::opticalflow::DenseOpticalFlow *> opticalFlows;
	cv::Mat referenceMask, currentMask;

public:
	explicit Setup() : VideoSetup() {
		stabilizer = std::make_shared<providentia::stabilization::SURFBFDynamicStabilization>();
		opticalFlows.emplace_back(new providentia::opticalflow::FarnebackDenseOpticalFlow());
		opticalFlows.emplace_back(new providentia::opticalflow::FarnebackDenseOpticalFlow());
		opticalFlows.emplace_back(new providentia::opticalflow::FarnebackDenseOpticalFlow());
//        stabilizer = std::make_shared<providentia::stabilization::ORBBFDynamicStabilization>();
//		stabilizer = std::make_shared<providentia::stabilization::FastFREAKBFDynamicStabilization>();
//        stabilizer->setShouldUpdateKeyframe(true);
	}

	cv::Mat createBackgroundSegmentationRow() {
		cv::cvtColor(cv::Mat(stabilizer->getSegmentor()->getBackgroundMask(frameGPU.size())), currentMask,
					 cv::COLOR_GRAY2BGR);
		cv::cvtColor(cv::Mat(stabilizer->getReferenceframeMask()), referenceMask, cv::COLOR_GRAY2BGR);
//		return ::hconcat(			{::MatofSize(frameGPU.size()), currentMask, referenceMask});
		return ::hconcat({::MatofSize(frameGPU.size()), currentMask});
	}

	cv::Mat createOpticalFlowRow() {
		opticalFlows[0]->calculate(frameGPU);
		opticalFlows[1]->calculate(stabilizer->getStabilizedFrame());
//		opticalFlows[2]->calculate(stabilizer->getReferenceframe());

//		return ::hconcat(			{opticalFlows[0]->draw(), opticalFlows[1]->draw(), opticalFlows[2]->draw()});
		return ::hconcat({
							 opticalFlows[0]->draw(),
							 opticalFlows[1]->draw()
						 });
	}

	void specificMainLoop() override {
		stabilizer->stabilize(frameGPU);
		totalAlgorithmsDuration = stabilizer->getTotalMilliseconds();
//		finalFrame = ::hconcat(			{stabilizer->getOriginalFrame(), stabilizer->getStabilizedFrame(), stabilizer->getReferenceframe()});

		finalFrame = ::hconcat({stabilizer->getOriginalFrame(), stabilizer->getStabilizedFrame()});
//		finalFrame = ::vconcat(			{finalFrame, createOpticalFlowRow(), createBackgroundSegmentationRow()});
		finalFrame = ::vconcat({finalFrame, createOpticalFlowRow()});

//        cv::imshow("Test", stabilizer->getMatcher()->draw());
	}

	void specificAddMessages()

	override {
		addRuntimeToFinalFrame("Feature detection", stabilizer->getFrameFeatureDetector()->getTotalMilliseconds(),
							   5, 20);
		addRuntimeToFinalFrame("Feature matching",
							   stabilizer->getMatcher()->getTotalMilliseconds(), 5, 40);
		addRuntimeToFinalFrame("Frame warping ",
							   stabilizer->getWarper()->getTotalMilliseconds(), 5, 60);
		addRuntimeToFinalFrame("Background segmentation",
							   stabilizer->getSegmentor()->getTotalMilliseconds(), 5, 80);
		addRuntimeToFinalFrame("Total stabilization", stabilizer->

								   getTotalMilliseconds(),

							   5, 100);
	}
};

int main(int argc, char const *argv[]) {
	Setup setup;
	setup.setRenderingScaleFactor(0.4);
	setup.mainLoop();
	return 0;
}
