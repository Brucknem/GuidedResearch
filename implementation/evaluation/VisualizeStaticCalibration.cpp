//
// Created by brucknem on 02.02.21.
//

#include <thread>
#include "Commons.hpp"
#include "Eigen/Dense"
#include "RenderingPipeline.hpp"
#include "WorldObjects.hpp"
#include "ObjectsLoading.hpp"
#include "CameraPoseEstimation.hpp"

using namespace providentia::evaluation;

/**
 * Setup to visualize the rendering pipeline.
 */
class Setup : public providentia::evaluation::ImageSetup {
public:
	/**
	 * The itnrinsics of the pinhole camera model.
	 */
	Eigen::Matrix<double, 3, 4> intrinsics = providentia::camera::getS40NCamFarIntrinsics();
	std::shared_ptr<providentia::calibration::CameraPoseEstimator> estimator;

	/**
	 * The [width, height] of the image.
	 */
	Eigen::Vector2i imageSize{1920, 1200};

	/**
	 * Some [x, y, z] translation of the camera in world space.
	 */
	Eigen::Vector3d initialTranslation{695825, 5.34608e+06, 546.63};
	Eigen::Vector3d translation = {initialTranslation};

	/**
	 * Some [x, y, z] euler angle rotation of the camera around the world axis
	 */
	Eigen::Vector3d initialRotation{85, 0, -20};
	Eigen::Vector3d rotation = {initialRotation};

	/**
	 * The trackbar translation values.
	 */
	int trackbarTranslationMiddle = 200;
	int trackbarTranslationX = trackbarTranslationMiddle;
	int trackbarTranslationY = trackbarTranslationMiddle;
	int trackbarTranslationZ = trackbarTranslationMiddle;

	/**
	 * The trackbar rotation values.
	 */
	int trackbarRotationMiddle = 100;
	int trackbarRotationX = trackbarRotationMiddle;
	int trackbarRotationY = trackbarRotationMiddle;
	int trackbarRotationZ = trackbarRotationMiddle;

	/**
	 * Flag for the background.
	 */
	int trackbarBackground = 4;

	int trackbarWeightScale = 100;

	int trackbarNumOutlier = 0;

	int trackbarShowIds = 0;

	/**
	 * The objects from the HD map.
	 */
	std::vector<providentia::calibration::WorldObject> objects;

	bool optimizationFinished = false;

	bool renderObjects = true;

	explicit Setup() : ImageSetup() {
		std::string pixelsFile;
		pixelsFile = "../misc/pixels.yaml";
//		pixelsFile = "../misc/pixels_min_num.yaml";
//		pixelsFile = "../misc/pixels_undetermined.yaml";
//		pixelsFile = "../misc/pixels_only_top_bottom.yaml";
//		pixelsFile = "../misc/pixels_only_bottom.yaml";
//		pixelsFile = "../misc/pixels_only_top.yaml";
		objects = providentia::calibration::LoadObjects("../misc/objects.yaml", pixelsFile,
														imageSize);

		estimator = std::make_shared<providentia::calibration::CameraPoseEstimator>(intrinsics, true,
																					std::pow(2, trackbarWeightScale));
//		estimator->setInitialGuess(initialTranslation, initialRotation);

	}

	void init() override {
		ImageSetup::init();

//		cv::createTrackbar("Translation X", windowName, &trackbarTranslationX, 2 * trackbarTranslationMiddle);
//		cv::createTrackbar("Translation Y", windowName, &trackbarTranslationY, 2 * trackbarTranslationMiddle);
//		cv::createTrackbar("Translation Z", windowName, &trackbarTranslationZ, 2 * trackbarTranslationMiddle);

//		cv::createTrackbar("Rotation X", windowName, &trackbarRotationX, 2 * trackbarRotationMiddle);
//		cv::createTrackbar("Rotation Y", windowName, &trackbarRotationY, 2 * trackbarRotationMiddle);
//		cv::createTrackbar("Rotation Z", windowName, &trackbarRotationZ, 2 * trackbarRotationMiddle);

		cv::createTrackbar("Background", windowName, &trackbarBackground, 10);
		cv::createTrackbar("Weight Scale", windowName, &trackbarWeightScale, 100);
//		cv::createTrackbar("Num Outliers", windowName, &trackbarNumOutlier, 100);
		cv::createTrackbar("Show Ids", windowName, &trackbarShowIds, 1);
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
	void specificMainLoop() override {
		finalFrame = frameCPU.clone();
		std::vector<cv::Mat> matChannels;
		cv::split(finalFrame, matChannels);
		// create alpha channel
		cv::Mat alpha = cv::Mat::ones(frameCPU.size(), CV_8UC1);
		matChannels.push_back(alpha);
		cv::merge(matChannels, finalFrame);
		finalFrame.convertTo(finalFrame, CV_64FC4, 1. / 255.);
		finalFrame = finalFrame * (trackbarBackground / 10.);

		if (!optimizationFinished) {
			optimizationFinished = estimator->isOptimizationFinished();
			if (optimizationFinished) {
				trackbarBackground = 4;
				cv::setTrackbarPos("Background", windowName, trackbarBackground);
			}
		}

//		cv::flip(finalFrame, finalFrame, 0);

		initialTranslation = estimator->getTranslation();
		initialRotation = estimator->getRotation();

		translation = {
			initialTranslation.x() + trackbarTranslationX - trackbarTranslationMiddle,
			initialTranslation.y() + trackbarTranslationY - trackbarTranslationMiddle,
			initialTranslation.z() + trackbarTranslationZ - trackbarTranslationMiddle,
		};

		rotation = {
			initialRotation.x() + (trackbarRotationX - trackbarRotationMiddle) / 10.,
			initialRotation.y() + (trackbarRotationY - trackbarRotationMiddle) / 10.,
			initialRotation.z() + (trackbarRotationZ - trackbarRotationMiddle) / 10.,
		};

		if (pressedKey == (int) 's') {
			renderObjects = !renderObjects;
		} else if (pressedKey == (int) 'e') {
			estimator->setWeightScale(std::pow(2, trackbarWeightScale));
			estimator->clearWorldObjects();
			estimator->addWorldObjects(objects);

//			for (int i = 0; i < trackbarNumOutlier; ++i) {
//				estimator->addWorldObject(
//					providentia::calibration::WorldObject(
//						providentia::calibration::ParametricPoint::OnLine(
//							{getRandom01() * imageSize.x(), getRandom01() * imageSize.y()},
//							Eigen::Vector3d{getRandom01(), getRandom01(), getRandom01()} * 1000,
//							{getRandom01(), getRandom01(), getRandom01()},
//							0
//						)
//					)
//				);
//			}

			optimizationFinished = false;
			estimator->estimateAsync(true);
		}
//
//		cv::rectangle(finalFrame, {0, finalFrame.rows - 68 - 24}, {600, finalFrame.rows}, {0, 0, 0}, -1);
//		cv::rectangle(finalFrame, {1345, finalFrame.rows - 190 - 24}, {finalFrame.cols, finalFrame.rows}, {0, 0, 0},
//					  -1);
//
//		addTextToFinalFrame("RED dots: Unmapped objects", 5, finalFrame.rows - 68);
//		addTextToFinalFrame("GREEN dots: Mapped objects", 5, finalFrame.rows - 44);

		std::stringstream ss;
		ss << *
			estimator;
		std::string line;
		int i = 0;
		while (
			getline(ss, line
			)) {
//			addTextToFinalFrame(line, 1350, finalFrame.rows - 190 + i++ * 24);
		}

		if (!renderObjects) {
			return;
		}

		for (
			const auto &worldObject: objects) {
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

};

int main(int argc, char const *argv[]) {
	Setup setup;
	setup.setRenderingScaleFactor(1);
//	setup.setOutputFolder("./stabilization/");
	setup.mainLoop();
	return 0;
}
