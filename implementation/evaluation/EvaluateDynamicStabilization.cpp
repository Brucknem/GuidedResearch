//
// Created by brucknem on 13.01.21.
//
#include <opencv2/cudawarping.hpp>
#include "DynamicStabilization.hpp"
#include "OpticalFlow.hpp"
#include "Commons.hpp"
#include "CSVWriter.hpp"

using namespace providentia::evaluation;

/**
 * Setup to visualize the total stabilization algorithm.
 */
class Setup : public ::VideoSetup {
private:
	/**
	 * The matcher used to match the features.
	 */
	providentia::stabilization::SURFBFDynamicStabilizer surf;
	providentia::stabilization::ORBBFDynamicStabilizer orb;
	providentia::stabilization::FastFREAKBFDynamicStabilizer fast;
	std::vector<providentia::opticalflow::FarnebackDenseOpticalFlow> opticalFlows;

	std::shared_ptr<::CSVWriter> csvWriter;
	int frameId = 0;
	bool withMask = false;

public:
	explicit Setup() : VideoSetup() {
		opticalFlows.emplace_back();
		opticalFlows.emplace_back();
		opticalFlows.emplace_back();
		opticalFlows.emplace_back();

		csvWriter = std::make_shared<::CSVWriter>("./evaluateDynamicStabilization.csv");
		*csvWriter << "FrameId" << "Original" << "SURF" << "ORB" << "FAST" << newline;
	}

	void calculateOpticalFlows() {
		if (withMask) {
			opticalFlows[0].calculate(frameGPU, surf.getBackgroundMask(frameGPU.size()));
			opticalFlows[1].calculate(surf.getStabilizedFrame(), surf.getBackgroundMask(frameGPU.size()));
			opticalFlows[2].calculate(orb.getStabilizedFrame(), orb.getBackgroundMask(frameGPU.size()));
			opticalFlows[3].calculate(fast.getStabilizedFrame(), fast.getBackgroundMask(frameGPU.size()));
		} else {
			opticalFlows[0].calculate(frameGPU);
			opticalFlows[1].calculate(surf.getStabilizedFrame());
			opticalFlows[2].calculate(orb.getStabilizedFrame());
			opticalFlows[3].calculate(fast.getStabilizedFrame());
		}
	}

	void specificMainLoop() override {
		surf.stabilize(frameGPU);
		orb.stabilize(frameGPU);
		fast.stabilize(frameGPU);

		calculateOpticalFlows();

		finalFrame = ::hconcat(
			{
				frameCPU,
				::addRuntimeToFrame(cv::Mat(surf.getStabilizedFrame()), "SURF + BF", surf.getTotalMilliseconds(), 2, 5,
									5),
				::addRuntimeToFrame(cv::Mat(orb.getStabilizedFrame()), "ORB + BF", orb.getTotalMilliseconds(), 2, 5,
									5),
				::addRuntimeToFrame(cv::Mat(fast.getStabilizedFrame()), "FAST + BF", fast.getTotalMilliseconds(), 2,
									5, 5)
			}
		);

		bufferCPU = ::hconcat(
			{
				::addText(opticalFlows[0].draw(),
						  "Mean pixel shift: " + std::to_string(opticalFlows[0].getMagnitudeMean()), 2, 5, 5),
				::addText(opticalFlows[1].draw(),
						  "Mean pixel shift: " + std::to_string(opticalFlows[1].getMagnitudeMean()), 2, 5, 5),
				::addText(opticalFlows[2].draw(),
						  "Mean pixel shift: " + std::to_string(opticalFlows[2].getMagnitudeMean()), 2, 5, 5),
				::addText(opticalFlows[3].draw(),
						  "Mean pixel shift: " + std::to_string(opticalFlows[3].getMagnitudeMean()), 2, 5, 5),
			}
		);
		finalFrame = ::vconcat({finalFrame, bufferCPU});

		if (withMask) {
			bufferCPU = ::hconcat(
				{
					::cvtColor(surf.getBackgroundMask(frameGPU.size()), cv::COLOR_GRAY2BGR),
					::cvtColor(surf.getBackgroundMask(frameGPU.size()), cv::COLOR_GRAY2BGR),
					::cvtColor(orb.getBackgroundMask(frameGPU.size()), cv::COLOR_GRAY2BGR),
					::cvtColor(fast.getBackgroundMask(frameGPU.size()), cv::COLOR_GRAY2BGR),
				}
			);
			finalFrame = ::vconcat({finalFrame, bufferCPU});
		}

		*csvWriter << frameId++;
		for (auto &opticalFlow: opticalFlows) {
			*csvWriter << opticalFlow.getMagnitudeMean();
		}
		*csvWriter << newline;
	}

	void specificAddMessages() override {
//		addRuntimeToFinalFrame("Feature detection",
//							   stabilizer->getFrameFeatureDetector()->getTotalMilliseconds(), 5, 20);
//		addRuntimeToFinalFrame("Feature matching",
//							   stabilizer->getMatcher()->getTotalMilliseconds(), 5, 40);
//		addRuntimeToFinalFrame("Frame warping ",
//							   stabilizer->getWarper()->getTotalMilliseconds(), 5, 60);
//		addRuntimeToFinalFrame("Background segmentation",
//							   stabilizer->getSegmentor()->getTotalMilliseconds(), 5, 80);
//		addRuntimeToFinalFrame("Total stabilization",
//							   stabilizer->getTotalMilliseconds(), 5, 100);
	}
};

int main(int argc, char const *argv[]) {
	Setup setup;
	setup.setRenderingScaleFactor(0.4);
	setup.mainLoop();
	return 0;
}
