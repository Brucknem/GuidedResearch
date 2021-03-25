//
// Created by brucknem on 02.02.21.
//

#include "Commons.hpp"
#include "Eigen/Dense"
#include "RenderingPipeline.hpp"
#include "WorldObjects.hpp"

using namespace providentia::evaluation;

/**
 * Setup to visualize the rendering pipeline.
 */
class Setup : public providentia::evaluation::ImageSetup {
public:
	/**
	 * The itnrinsics of the pinhole camera model.
	 */
	Eigen::Matrix<double, 3, 4> intrinsics = providentia::camera::getBlenderCameraIntrinsics();

	/**
	 * The [width, height] of the image.
	 */
	Eigen::Vector2d imageSize{1920, 1200};

	/**
	 * Some [x, y, z] translation of the camera in world space.
	 */
	Eigen::Vector3d translation{0, -10, 5};

	/**
	 * Some [x, y, z] euler angle rotation of the camera around the world axis
	 */
	Eigen::Vector3d rotation{90, 0, 0};

	explicit Setup() : ImageSetup() {}

	void renderScene() {
		for (int i = -7; i <= 7; i += 2) {
			for (int j = 0; j < 200; j += 5) {
				render(providentia::calibration::ParametricPoint::OnPlane(
					Eigen::Vector3d{0, 0, 0}, Eigen::Vector3d{1, 0, 0},
					Eigen::Vector3d{0, 1, 0}, i, j)
						   .getPosition(),
					   {1, 1, 1});
			}
		}

		for (int i = 0; i < 40; ++i) {
			for (int j = 0; j < 5; ++j) {
				render(providentia::calibration::ParametricPoint::OnLine(
					Eigen::Vector3d{-7.5, (double) i * 5., 0},
					Eigen::Vector3d{0, 0, 1}, j)
						   .getPosition(),
					   {1, 1, 0});
				render(providentia::calibration::ParametricPoint::OnLine(
					Eigen::Vector3d{7.5, (double) i * 5., 0},
					Eigen::Vector3d{0, 0, 1}, j)
						   .getPosition(),
					   {0, 1, 1});
			}
		}
	}

	void render(Eigen::Vector3d vector, const cv::Vec3d &color) {
		render(vector.x(), vector.y(), vector.z(), color);
	}

	void render(double x, double y, double z, const cv::Vec3d &color) {
		providentia::camera::render(translation, rotation,
									intrinsics, Eigen::Vector4d(x, y, z, 1), color,
									finalFrame);
	}

	void renderRandom(int amount) {
		for (int z = 0; z < amount; ++z) {
			render(getRandom01() * 20 - 10, getRandom01() * 20,
				   getRandom01() * 20 - 5,
				   {getRandom01(), getRandom01(), getRandom01()});
		}
	}

	void renderRegularGrid(int size, const cv::Vec3d &color = {-1, -1, -1}) {
		for (int z = -size / 2 + 5; z <= size / 2 + 5; ++z) {
			for (int y = 0; y <= size; ++y) {
				for (int x = -size / 2; x <= size / 2; ++x) {
					cv::Vec3d _color = color;
					if (_color[0] < 0) {
						_color = {getRandom01(), getRandom01(), getRandom01()};
					}
					render(x, y, z, _color);
				}
			}
		}
	}

	void renderOutlines() { // Vertical
		for (int i = 0; i <= 10; ++i) {
			render(-8, 0, i, {1, 0, 1});
			render(0, 0, i, {1, 1, 0});
			render(7.99, 0, i, {0, 1, 1});
		}

		// Horizontal
		for (int i = 0; i <= 16; ++i) {
			render(i - 8, 0, 0, {1, 1, 1});
			render(i - 8, 0, 5, {0, 0, 1});
			render(i - 8, 0, 9.99, {0, 1, 0});
		}
	}

protected:
	void specificMainLoop() override {
		finalFrame = cv::Mat::zeros(cv::Size(imageSize[0], imageSize[1]), CV_64FC4);

		renderScene();
		//		renderOutlines();
		//		renderRegularGrid(15);
		//		renderRandom(1000);
	}
};

int main(int argc, char const *argv[]) {
	Setup setup;
	setup.setRenderingScaleFactor(1);
	setup.mainLoop();
	return 0;
}
