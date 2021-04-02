//
// Created by brucknem on 13.01.21.
//
#include <opencv2/cudawarping.hpp>
#include "DynamicStabilization.hpp"
#include "OpticalFlow.hpp"
#include "Commons.hpp"
#include "CSVWriter.hpp"
#include <boost/filesystem/convenience.hpp>

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
	int padding = 10;
	boost::filesystem::path evaluationPath;
	bool writeBadFrames = false;

public:
	explicit Setup() : VideoSetup() {
		opticalFlows.emplace_back();
		opticalFlows.emplace_back();
		opticalFlows.emplace_back();
		opticalFlows.emplace_back();
	}

	void init() override {
		VideoSetup::init();

		evaluationPath =
			outputFolder / "DynamicStabilization" / boost::filesystem::path(inputResource).filename().string();
		if (!boost::filesystem::is_directory(evaluationPath)) {
			boost::filesystem::create_directories(evaluationPath);
		}

		csvWriter = std::make_shared<::CSVWriter>(evaluationPath / (boost::filesystem::path(inputResource).filename()
																		.string() + ".csv"));
		*csvWriter << "Frame" << "Original" << "SURF" << "ORB" << "FAST" << newline;
	}

	void calculateOpticalFlows() {
		if (withMask) {
			auto size = pad(frameGPU, padding).size();
			opticalFlows[0].calculate(pad(frameGPU, padding), surf.getBackgroundMask(size));
			opticalFlows[1].calculate(pad(surf.getStabilizedFrame(), padding), surf.getBackgroundMask(size));
			opticalFlows[2].calculate(pad(orb.getStabilizedFrame(), padding), orb.getBackgroundMask(size));
			opticalFlows[3].calculate(pad(fast.getStabilizedFrame(), padding), fast.getBackgroundMask(size));
		} else {
			opticalFlows[0].calculate(pad(frameGPU, padding));
			opticalFlows[1].calculate(pad(surf.getStabilizedFrame(), padding));
			opticalFlows[2].calculate(pad(orb.getStabilizedFrame(), padding));
			opticalFlows[3].calculate(pad(fast.getStabilizedFrame(), padding));

		}
	}

	void specificMainLoop() override {
		surf.stabilize(frameGPU);
		orb.stabilize(frameGPU);
		fast.stabilize(frameGPU);

		calculateOpticalFlows();

		finalFrame = ::hconcat(
			{
				::addText(frameCPU, "Frame: " + std::to_string(frameId), 2, 5, 5),
				::addRuntimeToFrame(cv::Mat(surf.getStabilizedFrame()), "SURF + BF", surf.getTotalMilliseconds(), 2, 5,
									5),
				::addRuntimeToFrame(cv::Mat(orb.getStabilizedFrame()), "ORB + BF", orb.getTotalMilliseconds(), 2, 5,
									5),
				::addRuntimeToFrame(cv::Mat(fast.getStabilizedFrame()), "FAST + BF", fast.getTotalMilliseconds(), 2,
									5, 5)
			}, padding
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
				}, padding
			);
			finalFrame = ::vconcat({finalFrame, bufferCPU});
		}

		double originalMagnitudeMean = opticalFlows[0].getMagnitudeMean();

		*csvWriter << frameId;
		bool write = false;
		std::string frameName = "frame_" + std::to_string(frameId);

		int flowNumber = 0;
		for (auto &opticalFlow: opticalFlows) {
			double currentMagnitudeMean = opticalFlow.getMagnitudeMean();
			*csvWriter << currentMagnitudeMean;

			if (originalMagnitudeMean - currentMagnitudeMean < 0) {
				frameName += "_";
				switch (flowNumber) {
					case 1:
						frameName += "surf";
						break;
					case 2:
						frameName += "orb";
						break;
					case 3:
						frameName += "fast";
						break;
					default:
						break;
				}
				write = true;
			}
			flowNumber++;
		}
		if (write && writeBadFrames) {
			cv::imwrite((evaluationPath / (frameName + ".png")).string(), finalFrame);
		}
		*csvWriter << newline;
		frameId++;
	}
};

int main(int argc, char const *argv[]) {
	Setup setup;
	setup.fromCLI(argc, argv);
	setup.setRenderingScaleFactor(0.4);
	setup.init();
	setup.mainLoop();
	return 0;
}
