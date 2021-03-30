//
// Created by brucknem on 30.03.21.
//

#include "ObjectTracking.hpp"
#include <opencv2/tracking.hpp>
#include <utility>

namespace providentia {
	namespace evaluation {
		const std::string ObjectTracking::TRACKER_TYPES[] = {"MIL", "KCF", "CSRT"};

		cv::Ptr<cv::Tracker> getTracker(const std::string &trackerType) {
			if (trackerType == "MIL")
				return cv::TrackerMIL::create();
			if (trackerType == "KCF") {
				return cv::TrackerKCF::create();
			}
			return cv::TrackerCSRT::create();
		}

		ObjectTracking::ObjectTracking(int trackerType, std::string trackerName, int y, cv::Scalar color)
			: trackerType(TRACKER_TYPES[trackerType]), tracker(getTracker(TRACKER_TYPES[trackerType])),
			  trackerName(std::move(trackerName)), y(y), color(color) {}

		void ObjectTracking::init(cv::Mat frame, const cv::Rect2d &_bbox) {
			bbox = _bbox;
			cv::Rect &tmpRect = bbox;
			tracker->init(frame, tmpRect);
			bbox = tmpRect;
		}

		bool ObjectTracking::isInitialized() {
			return !bbox.empty();
		}

		cv::Point2d ObjectTracking::getLowerMidpoint() const {
			return {bbox.x + 0.5 * bbox.width, static_cast<double>(bbox.y + bbox.height)};
		}

		cv::Point2d ObjectTracking::getMidpoint() const {
			return {bbox.x + 0.5 * bbox.width, bbox.y + 0.5 * bbox.height};
		}

		cv::Mat ObjectTracking::draw(const cv::Mat &frame) {
			std::string message = trackerName + ": ";

			cv::Mat result = frame.clone();
			if (!isInitialized()) {
				message += "Not initialized.";
			} else {
				if (trackingSuccessful) {
					cv::rectangle(result, bbox, color, 2, 1);

					cv::Point2d midpoint = getMidpoint();
					cv::circle(result, midpoint, 5, color, -1);
					message += "Tracking successful. [" +
							   std::to_string(midpoint.x) + ", " +
							   std::to_string(midpoint.y)
							   + "]";
				} else {
					message += "Tracking failed.";
				}
			}
			addText(result, message, 2, 5, y);
			return result;
		}

		void ObjectTracking::track(const cv::Mat &frame) {
			cv::Rect &tmpRect = bbox;
			trackingSuccessful = tracker->update(frame, tmpRect);
			bbox = tmpRect;

		}

		const cv::Rect &ObjectTracking::getBbox() const {
			return bbox;
		}

		bool ObjectTracking::isTrackingSuccessful() const {
			return trackingSuccessful;
		}
	}
}