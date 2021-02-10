//
// Created by brucknem on 13.01.21.
//
#include "TemplateMatcher.hpp"

int main(int argc, char const *argv[]) {
	auto templateMatcher = providentia::calibration::TemplateMatcher(
			"../misc/s40_n_cam_far_calibration_test_image.png",
			"../misc/leitpfosten_cropped.png"
	);
	templateMatcher.run();

	return 0;
}
