//
// Created by brucknem on 10.02.21.
//

#ifndef CAMERASTABILIZATION_TEMPLATEMATCHER_H
#define CAMERASTABILIZATION_TEMPLATEMATCHER_H

#include <string>
#include <utility>
#include "opencv2/opencv.hpp"
#include "FeatureDetection.hpp"
#include "FeatureMatching.hpp"

namespace providentia {
	namespace calibration {
		class TemplateMatcher {
		private:
			std::string windowName = "Template matcher";

			std::string filename, templateFilename;

			cv::Mat image, imageGray, templateImage;
			cv::TemplateMatchModes matchMode = cv::TM_CCOEFF_NORMED;

			cv::Mat matchingResult;
			std::vector<cv::Rect> matchRects;

		public:
			TemplateMatcher(std::string _filename, std::string _templateFilename) {
				filename = std::move(_filename);
				templateFilename = std::move(_templateFilename);

				cv::Rect roi = {cv::Point{10, 365}, cv::Point{25, 400}};
				image = cv::imread(filename)(roi);
//				cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
				cv::resize(image, image, roi.size() * 20);

				templateImage = cv::imread(templateFilename);

				cv::resize(templateImage, templateImage, roi.size() * 20);
//				cv::cvtColor(templateImage, templateImage, cv::COLOR_BGR2GRAY);
//				matchingResult = {image.size(), CV_32FC1};
			}

			void run() const {
				for (;;) {
					char c = (char) cv::waitKey(1);
					if (c == 'q') {
						break;
					}
					if (c == 'm') {
						providentia::features::SURFFeatureDetector frameDetector(100);
						providentia::features::SURFFeatureDetector templateDetector;

						frameDetector.detect(image);
						templateDetector.detect(templateImage);

						providentia::features::BruteForceFeatureMatcher matcher(cv::NORM_L2);
						matcher.setShouldUseFundamentalMatrix(false);
						matcher.match(std::make_shared<providentia::features::SURFFeatureDetector>(frameDetector),
									  std::make_shared<providentia::features::SURFFeatureDetector>(templateDetector));

						cv::imshow("frame", frameDetector.draw());
						cv::imshow("template", templateDetector.draw());
						cv::imshow("match", matcher.draw());
					}
					cv::imshow(windowName, draw());
				}
			}

			cv::Mat draw() const {
				cv::Mat result;
				result = image;
				return result;
			}
		};
	}
}

#endif //CAMERASTABILIZATION_TEMPLATEMATCHER_H
