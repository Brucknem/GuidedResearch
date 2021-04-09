//
// Created by brucknem on 02.02.21.
//

#include <thread>
#include <boost/algorithm/string/split.hpp>
#include "Commons.hpp"
#include "Eigen/Dense"
#include "RenderingPipeline.hpp"
#include "WorldObjects.hpp"
#include "ObjectsLoading.hpp"
#include "CameraPoseEstimation.hpp"
#include "CSVWriter.hpp"

#include <boost/foreach.hpp>
#include <boost/algorithm/string/trim.hpp>

using namespace providentia::evaluation;

/**
 * Setup to visualize the rendering pipeline.
 */
class Setup : public providentia::evaluation::ImageSetup {
public:
	/**
	 * The itnrinsics of the pinhole camera model.
	 */
	Eigen::Matrix<double, 3, 4> intrinsics;
	std::shared_ptr<providentia::calibration::CameraPoseEstimator> estimator;

	/**
	 * The [width, height] of the image.
	 */
	Eigen::Vector2i imageSize{1920, 1200};

	/**
	 * Some [x, y, z] translation of the camera in world space.
	 */
	Eigen::Vector3d initialTranslation{
		695962.92502753110602498055, 5346500.60284730326384305954 - 1000, 538.63933146729164036515
	};
	Eigen::Vector3d translation = {initialTranslation};

	/**
	 * Some [x, y, z] euler angle rotation of the camera around the world axis
	 */
	Eigen::Vector3d initialRotation{85, 0, -160};
	Eigen::Vector3d rotation = {initialRotation};

	/**
	 * Flag for the background.
	 */
	int trackbarBackground = 4;

	int trackbarShowIds = 0;

	int evaluationRun = -1;
	int weightScale = 100;
	int maxWeightScale = 100;

	int lambdaIndex = -1;
	std::vector<double> lambdaOptions{2., 4., 5., 6.};

	int rotationIndex = 0;
	std::vector<double> rotationOptions{0., 1., 5., 10., 50., 100., 500., 1000.};

	int runsPerScale = 100;

	/**
	 * The objects from the HD map.
	 */
	std::vector<providentia::calibration::WorldObject> objects;

	bool optimizationFinished = false;

	bool renderObjects = true;

	std::string pixelsFile, objectsFile;
	boost::filesystem::path evaluationPath;
	CSVWriter *extrinsicParametersWriter;
	CSVWriter *optimizationRunWriter;

	explicit Setup() : ImageSetup() {}

	boost::program_options::variables_map fromCLI(int argc, const char **argv) override {
		auto vm = ImageSetup::fromCLI(argc, argv);
		if (vm.count("intrinsics") <= 0) {
			std::cout << "Provide intrinsics.";
			exit(EXIT_FAILURE);
		}
		std::vector<std::string> rawIntrinsics;
		boost::split(rawIntrinsics, vm["intrinsics"].as<std::string>(), [](char c) { return c == ','; });
		if (rawIntrinsics.size() != 9) {
			std::cout << "Provide intrinsics as comma separated list in format \"f_x,0,c_x,0,f_y,c_y,0,skew,1\".";
			exit(EXIT_FAILURE);
		}
		std::vector<double> intrinsicsCasted;
		BOOST_FOREACH(std::string value, rawIntrinsics) {
						intrinsicsCasted.emplace_back(boost::lexical_cast<double>(boost::trim_copy(value)));
					};
		intrinsics = providentia::camera::getIntrinsicsMatrixFromConfig(intrinsicsCasted.data());

//		initialRotation.z() = vm["z_init"].as<int>();
		return vm;
	}

	void addAdditionalOptions(po::options_description *desc) override {
		desc->add_options()
			("intrinsics,t", po::value<std::string>(),
			 "The intrinsic parameters of the camera as comma separated list in format \"f_x,0,c_x,0,f_y,c_y,0,skew,"
			 "1\".");
//			("z_init,z", po::value<int>()->default_value(0),
//			 "The initial z rotation.");
	}

	void initCSVWriters() {
		evaluationPath =
			outputFolder / "StaticCalibration" / boost::filesystem::path(pixelsFile).parent_path().filename();
		if (!boost::filesystem::is_directory(evaluationPath)) {
			boost::filesystem::create_directories(evaluationPath);
		}
		auto suffix = getNowSuffix();
		extrinsicParametersWriter = new CSVWriter(evaluationPath / ("parameters" + suffix + ".csv"));

		*extrinsicParametersWriter << "Run"
								   << "Correspondences"
								   << "Penalize Scale [Lambdas]"
								   << "Penalize Scale [Rotation]"
								   << "Penalize Scale [Weights]"
								   << "Loss"
								   << "Loss [Correspondences]"
								   << "Loss [Lambdas]"
								   << "Loss [Rotations]"
								   << "Loss [Weights]"
								   << "Translation [x]"
								   << "Translation [y]"
								   << "Translation [z]"
								   << "Rotation [x]"
								   << "Rotation [y]"
								   << "Rotation [z]"
								   << "Weights [Avg]"
								   << "Weights [Min]"
								   << "Weights [Max]"
								   << newline
								   << flush;
	}

	void init() override {
		ImageSetup::init();

		pixelsFile = inputResource + ".yaml";
		objectsFile = (boost::filesystem::path(inputResource).parent_path() / "objects.yaml").string();

		objects = providentia::calibration::LoadObjects(objectsFile, pixelsFile, imageSize);
		estimator = std::make_shared<providentia::calibration::CameraPoseEstimator>(intrinsics, true,
																					powBase2(weightScale));

		if (!dontRenderFinalFrame) {
			cv::createTrackbar("Background", windowName, &trackbarBackground, 10);
			cv::createTrackbar("Show Ids", windowName, &trackbarShowIds, 1);
		}
		initCSVWriters();
	}

	void render(std::string id, Eigen::Vector3d vector, const cv::Vec3d &color, bool showId) {
		render(id, vector.x(), vector.y(), vector.z(), color, showId);
	}

	void render(std::string id, double x, double y, double z, const cv::Vec3d &color, bool showId) {
		Eigen::Vector4d vector{x, y, z, 1};
		Eigen::Vector4d vectorInCameraSpace = providentia::camera::toCameraSpace(
			translation.data(), rotation.data(), vector.data());

		if (std::abs(vectorInCameraSpace.z()) > 2000) {
			return;
		}

		bool flipped;
		auto pixel = providentia::camera::render(translation, rotation,
												 intrinsics, vector, color,
												 finalFrame, flipped);

		if (flipped) {
			return;
		}

//		if (pixel.x() >= 0 && pixel.x() < imageSize[0] &&
//			pixel.y() >= 0 && pixel.y() < imageSize[1]) {
//			std::cout << id << std::endl;
//		}

		std::stringstream ss;
		ss << std::fixed;
		ss << id;
//		ss << ": " << x << "," << y << "," << z;

		if (showId && trackbarShowIds == 1) {
			addTextToFinalFrame(ss.str(), pixel.x(), imageSize[1] - 1 - pixel.y());
		}
	}

protected:
	void initFinalFrame() {
		finalFrame = frameCPU.clone();
		std::vector<cv::Mat> matChannels;
		cv::split(finalFrame, matChannels);
		// create alpha channel
		cv::Mat alpha = cv::Mat::ones(frameCPU.size(), CV_8UC1);
		matChannels.push_back(alpha);
		cv::merge(matChannels, finalFrame);
		finalFrame.convertTo(finalFrame, CV_64FC4, 1. / 255.);
		finalFrame = finalFrame * (trackbarBackground / 10.);
	}

	void renderText() {
		cv::rectangle(finalFrame, {0, finalFrame.rows - 125}, {600, finalFrame.rows}, {0, 0, 0}, -1);
		cv::rectangle(finalFrame, {1345, finalFrame.rows - 190 - 24}, {finalFrame.cols, finalFrame.rows}, {0, 0, 0},
					  -1);

		addTextToFinalFrame("RED dots: Unmapped objects", 5, finalFrame.rows - 120);
		addTextToFinalFrame("GREEN dots: Mapped objects", 5, finalFrame.rows - 100);
		addTextToFinalFrame("Weight Scale: " + std::to_string(powBase2(weightScale)), 5, finalFrame.rows - 80);
		addTextToFinalFrame(
			"Run: " + std::to_string(evaluationRun + (lambdaIndex * runsPerScale) + ((lambdaOptions.size() *
																					  runsPerScale) * rotationIndex)) +
			"/"
			+ std::to_string(lambdaOptions.size() * rotationOptions.size() * runsPerScale), 5, finalFrame.rows - 60);
		addTextToFinalFrame("Lambda: "
							+ std::to_string(lambdaOptions[lambdaIndex]), 5, finalFrame.rows - 40);
		addTextToFinalFrame("Rotation: "
							+ std::to_string(rotationOptions[rotationIndex]), 5, finalFrame.rows - 20);

		std::stringstream ss;
		ss << *estimator;
		std::string line;
		int i = 0;
		while (getline(ss, line)) {
			addTextToFinalFrame(line, 1350, finalFrame.rows - 190 + i++ * 24);
		}
	}

	void render() {
		for (const auto &worldObject: objects) {
			bool idShown = false;
			for (const auto &point : worldObject.getPoints()) {
				Eigen::Vector3d p = point->getPosition();
				cv::Vec3d color = {0, 0, 1};
				if (point->hasExpectedPixel()) {
					color = {0, 1, 0};
				}
				render(worldObject.getId(), p, color, !idShown);
				idShown = true;
			}
		}
	}

	double powBase2(double val) {
		return std::pow(2, val);
	}

	void writeToCSV() {
		if (evaluationRun > -1) {
			auto weights = estimator->getWeights();

			double min_w = 1e100;
			double max_w = -1e100;
			double sum_w = 0;
			for (const auto &weight : weights) {
				if (weight < min_w) {
					min_w = weight;
				}
				if (weight > max_w) {
					max_w = weight;
				}
				sum_w += weight;
			}

			*extrinsicParametersWriter << evaluationRun
									   << (int) weights.size()
									   << lambdaOptions[lambdaIndex]
									   << rotationOptions[rotationIndex]
									   << (powBase2(weightScale))
									   << estimator->evaluate()
									   << estimator->evaluateCorrespondenceResiduals()
									   << estimator->evaluateLambdaResiduals()
									   << estimator->evaluateRotationResiduals()
									   << estimator->evaluateWeightResiduals()
									   << translation
									   << rotation
									   << sum_w / weights.size()
									   << min_w
									   << max_w
									   << newline;
		}
	}

	void doEvaluationIncrementations() {
		if (evaluationRun % runsPerScale == 0) {
			evaluationRun = 0;
			lambdaIndex++;
			if (lambdaIndex >= lambdaOptions.size()) {
				lambdaIndex = 0;
				rotationIndex++;
				if (rotationIndex >= rotationOptions.size()) {
					exit(EXIT_SUCCESS);
				}
			}
		}
	}

	void specificMainLoop() override {
		initFinalFrame();

		optimizationFinished = estimator->isOptimizationFinished();
		translation = estimator->getTranslation();
		rotation = estimator->getRotation();

		if (optimizationFinished) {
			writeToCSV();
			evaluationRun++;
			doEvaluationIncrementations();

			estimator->setWeightScale(powBase2(weightScale));
			estimator->setLambdaScale(lambdaOptions[lambdaIndex]);
			estimator->setRotationScale(rotationOptions[rotationIndex]);

			estimator->clearWorldObjects();
			estimator->addWorldObjects(objects);
//			estimator->guessRotation(initialRotation);
//			estimator->guessTranslation(initialTranslation);

			optimizationFinished = false;
			estimator->estimateAsync(true);
		}

		render();
		renderText();
	}
};

int main(int argc, char const *argv[]) {
	Setup setup;
	setup.fromCLI(argc, argv);
	setup.setRenderingScaleFactor(1);
//	setup.setOutputFolder("./stabilization/");
	setup.mainLoop();
	return 0;
}
