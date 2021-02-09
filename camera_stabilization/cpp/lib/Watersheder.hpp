//
// Created by brucknem on 09.02.21.
//

#ifndef CAMERASTABILIZATION_WATERSHEDER_HPP
#define CAMERASTABILIZATION_WATERSHEDER_HPP

#include <utility>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"

namespace providentia {
	namespace calibration {
		class Watersheder {
		private:
			int outputFileIndex = 0;

		public:
			std::string filename;
			cv::Mat image, imageGray, markerMask, watershedMask;
			std::string mainWindowName = "Watersheder";
			std::string resultWindowName = "Watersheder Result";
			cv::Point prevPt{-1, -1};

			bool modifierPressed = false;

			int posX = 0;
			int posY = 0;
			int width;
			int height;
			int thickness = 5;

			int getMaxX() {
				return image.cols - 1;
			}

			int getMaxY() {
				return image.rows - 1;
			}

			static void onMouse(int event, int x, int y, int flags, void *_watersheder) {
				auto watersheder = (Watersheder *) _watersheder;
				int totalX = watersheder->posX + x;
				int totalY = watersheder->posY + y;
				if (totalX < 0 || totalX >= watersheder->getMaxX() || totalY < 0 || y >= watersheder->getMaxY()) {
					return;
				}
				if (event == cv::EVENT_LBUTTONUP || !(flags & cv::EVENT_FLAG_LBUTTON)) {
					/// End of drawing
					watersheder->prevPt = cv::Point(-1, -1);
				} else if (event == cv::EVENT_LBUTTONDOWN) {
					/// Start of drawing
					watersheder->prevPt = cv::Point(totalX, totalY);
				} else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)) {
					/// Drawing
					cv::Point pt(totalX, totalY);
					if (watersheder->prevPt.x < 0) {
						watersheder->prevPt = pt;
					}
					cv::Scalar result = cv::Scalar::all(255);
					if (watersheder->modifierPressed) {
						result = cv::Scalar::all(0);
					}
					cv::line(watersheder->markerMask, watersheder->prevPt, pt, result, watersheder->thickness, 8, 0);
					watersheder->prevPt = pt;
				}
			}

			explicit Watersheder(const std::string &_filename) {
				filename = _filename;

				image = cv::imread(filename);
				cv::namedWindow(mainWindowName, cv::WINDOW_NORMAL);

				width = getMaxX();
				height = getMaxY();
				cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
				cv::cvtColor(imageGray, imageGray, cv::COLOR_GRAY2BGR);
				markerMask = cv::Mat(image.size(), CV_8UC3, cv::Scalar::all(0));
				watershedMask = cv::Mat(image.size(), CV_8UC3, cv::Scalar::all(0));

				cv::setMouseCallback(mainWindowName, onMouse, this);
				cv::createTrackbar("Pos X: ", mainWindowName, &posX, getMaxX());
				cv::createTrackbar("Pos Y: ", mainWindowName, &posY, getMaxY());

				cv::createTrackbar("Width: ", mainWindowName, &width, getMaxX());
				cv::createTrackbar("Height: ", mainWindowName, &height, getMaxY());

				cv::createTrackbar("Thickness: ", mainWindowName, &thickness, 10);

			}

			std::string createOutputFilename() {
				std::string segment;
				std::vector<std::string> seglist;
				std::stringstream ss(filename);
				while (std::getline(ss, segment, '/')) {
					seglist.push_back(segment);
				}
				auto file = seglist.back();
				return std::to_string(outputFileIndex++) + "_watershed_" + file;
//				seglist[seglist.size() - 1] = std::to_string(outputFileIndex++) + "_watershed_" + file;
//
//				ss.clear();
//				for (const auto &seg : seglist) {
//					ss << seg << "/";
//				}
//				auto result = ss.str();
//				result = result.substr(0, result.length() - 1);
//				return result;
			}

			void run() {
				for (;;) {
					char c = (char) cv::waitKey(1);
					if (c == 'q') {
						break;
					}
					if (c == 'r') {
						cv::destroyWindow(resultWindowName);
					}
					if (c == 'd') {
						modifierPressed = !modifierPressed;
					}
					if (c == 'c') {
						markerMask(getRoi()) = cv::Scalar::all(0);
					}
					if (c == 's') {
						cv::imwrite(createOutputFilename(), watershedMask);
					}
					if (c == 'n') {
						width = 50;
						height = 50;
					}
					if (c == 'b') {
						width = getMaxX();
						height = getMaxY();
					}
					if (c == 'w' || c == ' ') {
						int i, j, compCount = 0;
						std::vector<std::vector<cv::Point> > contours;
						std::vector<cv::Vec4i> hierarchy;
						cv::Mat _markerMask;
						cv::cvtColor(markerMask, _markerMask, cv::COLOR_BGR2GRAY);
						findContours(_markerMask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
						if (contours.empty()) {
							continue;
						}
						cv::Mat markers(markerMask.size(), CV_32S);
						markers = cv::Scalar::all(0);
						int idx = 0;
						for (; idx >= 0; idx = hierarchy[idx][0], compCount++) {
							drawContours(markers, contours, idx, cv::Scalar::all(compCount + 1), -1, 8, hierarchy,
										 INT_MAX);
						}

						if (compCount == 0) {
							continue;
						}
						std::vector<cv::Vec3b> colorTab;
						for (i = 0; i < compCount; i++) {
							int b = cv::theRNG().uniform(0, 255);
							int g = cv::theRNG().uniform(0, 255);
							int r = cv::theRNG().uniform(0, 255);
							colorTab.emplace_back((uchar) b, (uchar) g, (uchar) r);
						}
						double t = (double) cv::getTickCount();
						watershed(image, markers);
						t = (double) cv::getTickCount() - t;
						printf("execution time = %gms\n", t * 1000. / cv::getTickFrequency());
						watershedMask = cv::Mat(markers.size(), CV_8UC3);
						// paint the watershed image
						for (i = 0; i < markers.rows; i++)
							for (j = 0; j < markers.cols; j++) {
								int index = markers.at<int>(i, j);
								if (index == -1)
									watershedMask.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
								else if (index <= 0 || index > compCount)
									watershedMask.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
								else
									watershedMask.at<cv::Vec3b>(i, j) = colorTab[index - 1];
							}
						watershedMask = watershedMask * 0.5 + imageGray * 0.5;
						cv::namedWindow(resultWindowName, cv::WINDOW_GUI_NORMAL);
						cv::imshow(resultWindowName, watershedMask);
					}
					cv::imshow(mainWindowName, draw());
				}
			}

			cv::Rect getRoi() {
				int _posX = std::min(posX, getMaxX() - 20);
				int _posY = std::min(posY, getMaxY() - 20);
				int _width = std::max(20, std::min(width, getMaxX() - _posX));
				int _height = std::max(20, std::min(height, getMaxY() - _posY));
				return {_posX, _posY, _width, _height};
			}

			cv::Mat draw() {
				auto roi = getRoi();
				cv::Mat result = image(roi) + markerMask(roi);
//				cv::bitwise_and(image(roi) + , result, result);
				return result.clone();
			};
		};
	}
}
#endif //CAMERASTABILIZATION_WATERSHEDER_HPP
