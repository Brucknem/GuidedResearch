//
// Created by brucknem on 13.01.21.
//
#include <opencv2/cudawarping.hpp>
#include "DynamicStabilization.hpp"
#include "Commons.hpp"
#include "Watersheder.hpp"

int main(int argc, char const *argv[]) {
	auto windowName = "Watersheder";
	cv::namedWindow(windowName, cv::WINDOW_OPENGL);
	auto watersheder = providentia::calibration::Watersheder(cv::imread(
			"../misc/s40_n_cam_far_calibration_test_image.png"), windowName
	);

	while (cv::waitKey(1) != 'q') {
		cv::imshow(windowName, watersheder.draw());
	}

	return 0;
}
