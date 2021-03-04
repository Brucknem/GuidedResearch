//
// Created by brucknem on 02.02.21.
//

#include "Commons.hpp"
#include "Eigen/Dense"
#include "RenderingPipeline.hpp"
#include "WorldObjects.hpp"

using namespace providentia::runnable;

/**
 * Setup to visualize the rendering pipeline.
 */
class Setup : public providentia::runnable::BaseSetup {
public:
	/**
	 * The [sensorWidth, aspectRatio, focalLength] of the pinhole camera model.
	 */
	Eigen::Matrix<double, 3, 4> intrinsics = providentia::camera::getIntrinsicsMatrixFromConfig(new double[9]{
		9023.482825, 0.000000, 1222.314303, 0.000000, 9014.504360, 557.541182, 0.000000, 0.000000, 1.000000}
	);

	/**
	 * The [width, height] of the image.
	 */
	Eigen::Vector2d imageSize{1920, 1200};

	/**
	 * Some [x, y, z] translation of the camera in world space.
	 */
	Eigen::Vector3d translation{0, 0, 5};

	/**
	 * Some [x, y, z] euler angle rotation of the camera around the world axis
	 */
	Eigen::Vector3d rotation{90, 0, 0};

	explicit Setup(int argc, char const *argv[]) : BaseSetup(argc, argv) {}

	void render(Eigen::Vector3d vector, const cv::Vec3d &color) {
		render(vector.x(), vector.y(), vector.z(), color);
	}

	void render(double x, double y, double z, const cv::Vec3d &color) {
		providentia::camera::render(translation, rotation,
									intrinsics, Eigen::Vector4d(x, y, z, 1), color,
									finalFrame);
	}

protected:
	void specificMainLoop() override {
		finalFrame = frameCPU;
	}
};

int main(int argc, char const *argv[]) {
	Setup setup(argc, argv);
	setup.setRenderingScaleFactor(1);
	setup.mainLoop();
	return 0;
}
