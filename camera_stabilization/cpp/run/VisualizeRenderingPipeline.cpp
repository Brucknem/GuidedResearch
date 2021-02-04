//
// Created by brucknem on 02.02.21.
//

#include "Commons.hpp"
#include "Eigen/Dense"
#include "RenderingPipeline.hpp"

/**
 * Setup to visualize the rendering pipeline.
 */
class Setup : public providentia::runnable::BaseSetup {
private:
	Eigen::Vector2d frustumParameters{1, 1000};
	Eigen::Vector3d intrinsics{32, 1920. / 1200., 20};

	Eigen::Vector3d rotation{90, 0, 0};
	Eigen::Vector3d translation{0, -10, 5};

public:
	explicit Setup(int argc, char const *argv[]) : BaseSetup(argc, argv) {
	}

	void render(double x, double y, double z, const cv::Vec3d &color) {
		providentia::camera::render(
				translation, rotation,
				frustumParameters, intrinsics,
				Eigen::Vector4d(x, y, z, 1),
				color, finalFrame);
	}

	void specificMainLoop() override {
		finalFrame = cv::Mat::zeros(cv::Size(1920, 1200), CV_64FC4);

		// Vertical
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
};

int main(int argc, char const *argv[]) {
	Setup setup(argc, argv);
	setup.setRenderingScaleFactor(1);
	setup.mainLoop();
	return 0;
}
