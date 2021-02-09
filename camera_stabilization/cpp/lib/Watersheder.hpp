//
// Created by brucknem on 09.02.21.
//

#ifndef CAMERASTABILIZATION_WATERSHEDER_HPP
#define CAMERASTABILIZATION_WATERSHEDER_HPP

#include <utility>

#include "opencv2/opencv.hpp"

namespace providentia {
	namespace calibration {
		class Watersheder {
		public:
			cv::Mat image;
			std::string windowName;

			int posX = 0;
			int posY = 0;
			int width;
			int height;

			int getMaxX() {
				return image.cols - 1;
			}

			int getMaxY() {
				return image.rows - 1;
			}

			explicit Watersheder(const cv::Mat &_image, std::string _windowName) {
				image = _image;
				windowName = std::move(_windowName);
				cv::createTrackbar("Pos X: ", windowName, &posX, getMaxX());
				cv::createTrackbar("Pos Y: ", windowName, &posY, getMaxY());

				cv::createTrackbar("Width: ", windowName, &width, getMaxX());
				cv::createTrackbar("Height: ", windowName, &height, getMaxY());

				width = getMaxX();
				height = getMaxY();
			}

			cv::Mat draw() {
				int _posX = std::min(posX, getMaxX() - 20);
				int _posY = std::min(posY, getMaxY() - 20);
				int _width = std::max(20, std::min(width, getMaxX() - _posX));
				int _height = std::max(20, std::min(height, getMaxY() - _posY));
				return image({_posX, _posY, _width, _height}).clone();
			};
		};
	}
}
#endif //CAMERASTABILIZATION_WATERSHEDER_HPP
