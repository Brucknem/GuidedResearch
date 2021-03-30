//
// Created by brucknem on 13.01.21.
//
#include <opencv2/cudawarping.hpp>
#include "DynamicStabilization.hpp"
#include "OpticalFlow.hpp"
#include "Commons.hpp"
#include "CSVWriter.hpp"
#include <boost/filesystem/convenience.hpp>
#include "ObjectTracking.hpp"

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

	std::vector<ObjectTracking> originalTrackers;
	std::vector<ObjectTracking> surfTrackers;
	std::vector<ObjectTracking> orbTrackers;
	std::vector<ObjectTracking> fastTrackers;

	std::vector<cv::Rect2d> originalBoundingBoxes{
		cv::Rect2d{540., 175., 100., 145.},
	};

	std::vector<::CSVWriter *> csvWriters;
	boost::filesystem::path evaluationPath;
	int frameId = 0;
	int trackerType = 2;
public:
	explicit Setup() : VideoSetup() {
		for (int i = 0; i < originalBoundingBoxes.size(); i++) {
			originalTrackers.emplace_back(trackerType, "Original", 5 + 24 * i);
			surfTrackers.emplace_back(trackerType, "SURF", 5 + 24 * i);
			orbTrackers.emplace_back(trackerType, "ORB", 5 + 24 * i);
			fastTrackers.emplace_back(trackerType, "FAST", 5 + 24 * i);
		}
	}

	void init() override {
		VideoSetup::init();

		evaluationPath =
			outputFolder / "ObjectTracking" / boost::filesystem::path(inputResource).filename().string();
		if (!boost::filesystem::is_directory(evaluationPath)) {
			boost::filesystem::create_directories(evaluationPath);
		}

		getNextFrame();

		for (int i = 0; i < originalBoundingBoxes.size(); i++) {
			originalTrackers[i].init(frameCPU, originalBoundingBoxes[i]);
			surfTrackers[i].init(frameCPU, originalBoundingBoxes[i]);
			orbTrackers[i].init(frameCPU, originalBoundingBoxes[i]);
			fastTrackers[i].init(frameCPU, originalBoundingBoxes[i]);

			csvWriters.emplace_back(new CSVWriter(evaluationPath /
												  (boost::filesystem::path(inputResource).filename().string() +
												   "_object_" +
												   std::to_string(i) + ".csv")));
			*csvWriters.back() << "Frame" <<
							   "Original [x]" << "Original [y]" << "Original [w]" << "Original [h]" << "Original [mx]"
							   << "Original [my]" <<
							   "SURF [x]" << "SURF [y]" << "SURF [w]" << "SURF [h]" << "SURF [mx]" << "SURF [my]" <<
							   "ORB [x]" << "ORB [y]" << "ORB [w]" << "ORB [h]" << "ORB [mx]" << "ORB [my]" <<
							   "FAST [x]" << "FAST [y]" << "FAST [w]" << "FAST [h]" << "FAST [mx]" << "FAST [my]" <<
							   newline;
		}
	}

	void addTrackingResult(CSVWriter *csvWriter, const ObjectTracking &tracker) {
		if (tracker.isTrackingSuccessful()) {
			*csvWriter << tracker.getBbox() << tracker.getMidpoint();
		} else {
			for (int i = 0; i < 6; i++) {
				*csvWriter << -1;
			}
		}
	}

	void specificMainLoop() override {
		surf.stabilize(frameGPU);
		orb.stabilize(frameGPU);
		fast.stabilize(frameGPU);

		std::vector<cv::Mat> resultFrames{
			frameCPU.clone(),
			cv::Mat(surf.getStabilizedFrame()),
			cv::Mat(orb.getStabilizedFrame()),
			cv::Mat(fast.getStabilizedFrame())
		};
		for (int i = 0; i < originalBoundingBoxes.size(); i++) {
			originalTrackers[i].track(resultFrames[0]);
			surfTrackers[i].track(resultFrames[1]);
			orbTrackers[i].track(resultFrames[2]);
			fastTrackers[i].track(resultFrames[3]);

			resultFrames[0] = originalTrackers[i].draw(resultFrames[0]);
			resultFrames[1] = surfTrackers[i].draw(resultFrames[1]);
			resultFrames[2] = orbTrackers[i].draw(resultFrames[2]);
			resultFrames[3] = fastTrackers[i].draw(resultFrames[3]);

			*csvWriters[i] << frameId;
			addTrackingResult(csvWriters[i], originalTrackers[i]);
			addTrackingResult(csvWriters[i], surfTrackers[i]);
			addTrackingResult(csvWriters[i], orbTrackers[i]);
			addTrackingResult(csvWriters[i], fastTrackers[i]);
			*csvWriters[i] << newline;
		}

		finalFrame = ::hconcat(
			{
				::addText(resultFrames[0], "Frame: " + std::to_string(frameId), 2, 5, frameCPU.rows - 50),
				resultFrames[1],
				resultFrames[2],
				resultFrames[3],
			}
		);


//		*csvWriter << newline;
		frameId++;
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
	setup.fromCLI(argc, argv);
	setup.setRenderingScaleFactor(0.4);
//	setup.setRenderingScaleFactor(1);
	setup.init();
	setup.mainLoop();
	return 0;
}
