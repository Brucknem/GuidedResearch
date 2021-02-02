//
// Created by brucknem on 02.02.21.
//

#include "Camera.hpp"
#include "Commons.hpp"

/**
 * Setup to visualize the rendering pipeline.
 */
class Setup : public providentia::runnable::BaseSetup {
private:
	/**
	 * The virtual camera.
	 */
	std::shared_ptr<providentia::camera::Camera> camera;

public:
	explicit Setup(int argc, char const *argv[]) : BaseSetup(argc, argv) {
		camera = std::make_shared<providentia::camera::BlenderCamera>();
	}

	void specificMainLoop() override {
		// Vertical
		for (int i = 0; i <= 10; ++i) {
			camera->render(-8, 0, i, {1, 0, 1});
			camera->render(0, 0, i, {1, 1, 0});
			camera->render(7.99, 0, i, {0, 1, 1});
		}

		// Horizontal
		for (int i = 0; i <= 16; ++i) {
			camera->render(i - 8, 0, 0, {1, 1, 1});
			camera->render(i - 8, 0, 5, {0, 0, 1});
			camera->render(i - 8, 0, 9.99, {0, 1, 0});
		}

		finalFrame = camera->getImage();
		camera->resetImage();
	}

};

int main(int argc, char const *argv[]) {
	Setup setup(argc, argv);
	setup.setRenderingScaleFactor(1);
	setup.mainLoop();
	return 0;
}
