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
#include "FrameWarping.hpp"
#include "Color.hpp"

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

	ObjectTracking tracking;

	std::vector<cv::Rect2d> originalBoundingBoxes;
	std::vector<cv::Scalar> boundingBoxColors;
	std::vector<std::string> objectNames;
	std::vector<std::string> stabilizerNames = {"Original", "SURF", "ORB", "FAST"};

	std::vector<::CSVWriter *> csvWriters;
	int frameId = 0;
	int trackerType = 2;
	int warmUp = 25;

public:
	explicit Setup() : VideoSetup() {}

	boost::program_options::variables_map fromCLI(int argc, const char **argv) override {
		auto vm = VideoSetup::fromCLI(argc, argv);

		cv::RNG rng(43);
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
			boundingBoxColors.emplace_back(Color::any(rng));
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
			tracking.addTracker(frameCPU, trackerType, originalBoundingBoxes[i], boundingBoxColors[i]);

			csvWriters.emplace_back(new CSVWriter(outputFolder / (objectNames[i] + ".csv")));
			*csvWriters.back() << "Frame";
			for (const auto &name : stabilizerNames) {
				TrackerWrapper::addHeader(csvWriters.back(), name);
			}
			*csvWriters.back() << newline;
		}
	}

	void specificMainLoop() override {
		bufferCPU = tracking.track(frameCPU);

		if (tracking.isTrackingLost()) {
			exit(EXIT_SUCCESS);
		}

		bufferGPU.upload(bufferCPU);

		surf.stabilize(frameGPU);
		orb.stabilize(frameGPU);
		fast.stabilize(frameGPU);

		std::vector<cv::Mat> homographies;
		homographies.emplace_back(cv::Mat::eye(3, 3, CV_64F));
		homographies.emplace_back(surf.getHomography());
		homographies.emplace_back(orb.getHomography());
		homographies.emplace_back(fast.getHomography());

		std::vector<cv::Mat> resultFrames;
		for (int i = 0; i < homographies.size(); i++) {
			resultFrames.emplace_back(
				::addText(
					(tracking * homographies[i]).draw(
						cv::Mat(providentia::stabilization::FrameWarper::warp(frameGPU, homographies[i]))
					), stabilizerNames[i] + " - Frame: " + std::to_string(frameId), 2, 5,
					frameCPU.rows - 50)
			);
		}

		if (frameId > warmUp) {
			for (auto csvWriter : csvWriters) {
				*csvWriter << frameId;
			}

			for (int j = 0; j < homographies.size(); j++) {
				auto warpedTracker = tracking * homographies[j];
				std::vector<cv::Point2d> midPoints = warpedTracker.getMidpoints();

				bufferCPU = resultFrames[j];
				for (int i = 0; i < midPoints.size(); i++) {
					std::stringstream ss;
					ss << "[";
					ss << std::setw(7) << midPoints[i].x;
					ss << ", ";
					ss << std::setw(7) << midPoints[i].y;
					ss << "]";
					bufferCPU = ::addText(bufferCPU, ss.str(), 2, 5, (int) (5. + frameCPU.rows * 0.05 * i),
										  boundingBoxColors[i]);

					*(csvWriters[i]) << warpedTracker.getTrackers()[i];
				}

				resultFrames[j] = bufferCPU;
			}

			for (auto csvWriter : csvWriters) {
				*csvWriter << newline;
			}
		}

		finalFrame = ::hconcat(
			{
				resultFrames[0],
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
