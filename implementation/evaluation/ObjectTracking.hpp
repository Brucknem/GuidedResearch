//
// Created by brucknem on 30.03.21.
//

#ifndef CAMERASTABILIZATION_OBJECTTRACKING_HPP
#define CAMERASTABILIZATION_OBJECTTRACKING_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include "Commons.hpp"

namespace providentia {
	namespace evaluation {
		class ObjectTracking {
			const static std::string TRACKER_TYPES[8];

			std::string trackerType;
			cv::Ptr<cv::Tracker> tracker;

			cv::Rect bbox;

			bool trackingSuccessful = false;

			std::string trackerName;

			int y;

			cv::Scalar color;

		public:

			explicit ObjectTracking(int trackerType, std::string trackerName, int y, cv::Scalar color);

			bool isInitialized();

			bool isTrackingSuccessful() const;

			void init(cv::Mat frame, const cv::Rect2d &bbox);

			void track(const cv::Mat &frame);

			cv::Mat draw(const cv::Mat &frame);

			const cv::Rect &getBbox() const;

			cv::Point2d getLowerMidpoint() const;

			cv::Point2d getMidpoint() const;
		};
	}
}

#endif //CAMERASTABILIZATION_OBJECTTRACKING_HPP
