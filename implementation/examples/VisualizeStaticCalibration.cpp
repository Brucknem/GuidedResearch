//
// Created by brucknem on 02.02.21.
//

#include "Commons.hpp"
#include "Eigen/Dense"
#include "RenderingPipeline.hpp"
#include "WorldObjects.hpp"
#include "ObjectsLoading.hpp"
#include "CameraPoseEstimation.hpp"

using namespace providentia::runnable;

/**
 * Setup to visualize the rendering pipeline.
 */
class Setup : public providentia::runnable::ImageSetup {
public:
	/**
	 * The itnrinsics of the pinhole camera model.
	 */
	Eigen::Matrix<double, 3, 4> intrinsics = providentia::camera::getS40NCamFarIntrinsics();

	/**
	 * The [width, height] of the image.
	 */
	Eigen::Vector2i imageSize{1920, 1200};

	/**
	 * Some [x, y, z] translation of the camera in world space.
	 */
	Eigen::Vector3d initialTranslation{695856.16249748704, 5346096.1679782663, 540.98973892807601};
	Eigen::Vector3d translation = {initialTranslation};

	/**
	 * Some [x, y, z] euler angle rotation of the camera around the world axis
	 */
	Eigen::Vector3d initialRotation{87, 0, -16.5};
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
	int trackbarBackground = 1;

	/**
	 * The objects from the HD map.
	 */
	std::vector<providentia::calibration::WorldObject> objects;

	explicit Setup() : ImageSetup() {
		objects = providentia::calibration::LoadObjects("../misc/objects.yaml", imageSize);

		providentia::calibration::CameraPoseEstimator estimator{intrinsics};
		estimator.setInitialGuess(initialTranslation, initialRotation);
		estimator.addWorldObjects(objects);
		estimator.estimate(true);
		initialTranslation = estimator.getTranslation();
		initialRotation = estimator.getRotation();

		cv::createTrackbar("Translation X", windowName, &trackbarTranslationX, 2 * trackbarTranslationMiddle);
		cv::createTrackbar("Translation Y", windowName, &trackbarTranslationY, 2 * trackbarTranslationMiddle);
		cv::createTrackbar("Translation Z", windowName, &trackbarTranslationZ, 2 * trackbarTranslationMiddle);

		cv::createTrackbar("Rotation X", windowName, &trackbarRotationX, 2 * trackbarRotationMiddle);
		cv::createTrackbar("Rotation Y", windowName, &trackbarRotationY, 2 * trackbarRotationMiddle);
		cv::createTrackbar("Rotation Z", windowName, &trackbarRotationZ, 2 * trackbarRotationMiddle);

		cv::createTrackbar("Background", windowName, &trackbarBackground, 1);
	}

	void render(std::string id, Eigen::Vector3d vector, const cv::Vec3d &color) {
		render(id, vector.x(), vector.y(), vector.z(), color);
	}

	void render(std::string id, double x, double y, double z, const cv::Vec3d &color) {
		bool flipped;
		auto pixel = providentia::camera::render(translation, rotation,
												 intrinsics, Eigen::Vector4d(x, y, z, 1), color,
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
		addTextToFinalFrame(ss.str(), pixel.x(), imageSize[1] - 1 - pixel.y());
	}

protected:
	void specificMainLoop() override {
		if (trackbarBackground == 1) {
			finalFrame = frameCPU.clone();
			std::vector<cv::Mat> matChannels;
			cv::split(finalFrame, matChannels);
			// create alpha channel
			cv::Mat alpha = cv::Mat::ones(frameCPU.size(), CV_8UC1);
			matChannels.push_back(alpha);
			cv::merge(matChannels, finalFrame);
			finalFrame.convertTo(finalFrame, CV_64FC4, 1. / 255.);
		} else {
			finalFrame = cv::Mat::zeros(cv::Size(imageSize[0], imageSize[1]), CV_64FC4);
		}


//		cv::flip(finalFrame, finalFrame, 0);

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

		for (const auto &worldObject : objects) {
			for (const auto &point : worldObject.getPoints()) {
				Eigen::Vector3d p = point->getPosition();
				cv::Vec3d color = {1, 1, 1};
				if (point->hasExpectedPixel()) {
					color = {0, 0, 1};
				}
				render(worldObject.getId(), p, color);
			}
		}
	}
};

int main(int argc, char const *argv[]) {
	Setup setup;
	setup.setRenderingScaleFactor(1);
	setup.mainLoop();
	return 0;
}
