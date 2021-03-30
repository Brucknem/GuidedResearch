//
// Created by brucknem on 13.01.21.
//
#include <opencv2/cudawarping.hpp>
#include "DynamicStabilization.hpp"
#include "Commons.hpp"
#include "CSVWriter.hpp"
#include <boost/filesystem/convenience.hpp>
#include "ObjectTracking.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>

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

	std::vector<cv::Rect2d> originalBoundingBoxes;
	std::vector<cv::Scalar> boundingBoxColors;
	std::vector<std::string> objectNames;

	std::vector<::CSVWriter *> csvWriters;
	int frameId = 0;
	int trackerType = 2;

	int lineHeight = 42;

public:
	explicit Setup() : VideoSetup() {}

	boost::program_options::variables_map fromCLI(int argc, const char **argv) override {
		auto vm = VideoSetup::fromCLI(argc, argv);

		cv::RNG rng(1253434493);
		std::vector<std::string> rawBboxes;
		boost::split(rawBboxes, vm["bboxes"].as<std::string>(), [](char c) { return c == ','; });
		boost::split(objectNames, vm["names"].as<std::string>(), [](char c) { return c == ','; });

		if (rawBboxes.size() % 4 != 0) {
			std::cout << "The bounding boxes have to be defined as packs of 4 [x, y, w, h] values." << std::endl;
			exit(EXIT_FAILURE);
		}
		if (rawBboxes.size() / 4 != objectNames.size()) {
			std::cout << "There must be given exactly the same number of names and bounding boxes." << std::endl;
			exit(EXIT_FAILURE);
		}

		std::vector<int> rawBboxValues;
		BOOST_FOREACH(std::string value, rawBboxes) { rawBboxValues.emplace_back(std::stoi(value)); };
		for (int i = 0; i < rawBboxes.size(); i += 4) {
			originalBoundingBoxes.emplace_back(
				rawBboxValues[i + 0], rawBboxValues[i + 1],
				rawBboxValues[i + 2], rawBboxValues[i + 3]
			);
		}

		for (int i = 0; i < originalBoundingBoxes.size(); i++) {
			boundingBoxColors.emplace_back(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255), 1);

			originalTrackers.emplace_back(trackerType, "Original", 5 + lineHeight * i, boundingBoxColors[i]);
			surfTrackers.emplace_back(trackerType, "SURF", 5 + lineHeight * i, boundingBoxColors[i]);
			orbTrackers.emplace_back(trackerType, "ORB", 5 + lineHeight * i, boundingBoxColors[i]);
			fastTrackers.emplace_back(trackerType, "FAST", 5 + lineHeight * i, boundingBoxColors[i]);
		}
		return vm;
	}

	void addAdditionalOptions(po::options_description *desc) override {
		desc->add_options()
			("bboxes,b", po::value<std::string>()->default_value(""), "The initial bounding boxes of the vehicles to "
																	  "track.")
			("names,n", po::value<std::string>()->default_value(""), "The names of the vehicles to track.");
	}

	void init() override {
		VideoSetup::init();

		outputFolder = outputFolder / "ObjectTracking" / boost::filesystem::path(inputResource).filename().string();
		if (!boost::filesystem::is_directory(outputFolder)) {
			boost::filesystem::create_directories(outputFolder);
		}

		getNextFrame();

		for (int i = 0; i < originalBoundingBoxes.size(); i++) {
			originalTrackers[i].init(frameCPU, originalBoundingBoxes[i]);
			surfTrackers[i].init(frameCPU, originalBoundingBoxes[i]);
			orbTrackers[i].init(frameCPU, originalBoundingBoxes[i]);
			fastTrackers[i].init(frameCPU, originalBoundingBoxes[i]);

			csvWriters.emplace_back(new CSVWriter(outputFolder / (objectNames[i] + ".csv")));
			*csvWriters.back() << "Frame" <<
							   "Original [x]" << "Original [y]" << "Original [w]" << "Original [h]" << "Original [mx]"
							   << "Original [my]" <<
							   "SURF [x]" << "SURF [y]" << "SURF [w]" << "SURF [h]" << "SURF [mx]" << "SURF [my]" <<
							   "ORB [x]" << "ORB [y]" << "ORB [w]" << "ORB [h]" << "ORB [mx]" << "ORB [my]" <<
							   "FAST [x]" << "FAST [y]" << "FAST [w]" << "FAST [h]" << "FAST [mx]" << "FAST [my]" <<
							   newline;
		}
	}

	static void addTrackingResult(CSVWriter *csvWriter, const ObjectTracking &tracker) {
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

		frameId++;
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
