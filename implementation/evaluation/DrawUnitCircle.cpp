//
// Created by brucknem on 12.04.21.
//

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#define PIXEL_RADIUS 500.
#define MAX_DISTANCE 2000.

int centimetersToPixel(int centimeters, double radius) {
	return std::round((PIXEL_RADIUS / MAX_DISTANCE) * centimeters);
}

int pixelsToCentimeters(double pixels, double radius) {
	return std::round((MAX_DISTANCE / radius) * pixels);
}

int main(int argc, const char **argv) {

	std::string windowName = "Unit Circle";
	cv::namedWindow(windowName);

	int degree = 0;
	int radius = 100;
	int distanceX = 0;
	int distanceY = 0;

	cv::createTrackbar("Degree", windowName, &degree, 360);
	cv::createTrackbar("Distance X", windowName, &distanceX, MAX_DISTANCE * 2);
	cv::createTrackbar("Distance Y", windowName, &distanceY, MAX_DISTANCE * 2);

	int pressedKey;
	while (true) {
		cv::Mat unitCircle = cv::Mat(1200, 1200, CV_8UC3, cv::Scalar(255, 255, 255));
		auto midPoint = cv::Point{unitCircle.cols / 2, unitCircle.rows / 2};

		cv::circle(unitCircle, midPoint, radius, cv::Scalar(0, 0, 0), 5);
		cv::line(unitCircle, midPoint, {midPoint.x, 0}, cv::Scalar(255, 100, 0), 3);
		cv::line(unitCircle, midPoint, {unitCircle.cols, midPoint.y}, cv::Scalar(255, 100, 0), 3);

		cv::Vec3d vector = {1, 0, 0};
		double rads = -(degree / 180.) * M_PI;
		cv::Affine3d rotationMatrix(cv::Vec3f{0, 0, 1} * rads, cv::Vec3f{0, 0, 0});
		vector = (rotationMatrix * vector) * RADIUS;

		cv::Point endPoint = {midPoint.x + (int) vector(0), midPoint.y + (int) vector(1)};
		cv::line(unitCircle, midPoint, endPoint, cv::Scalar(0, 255, 0), 5);

		cv::Point2d xEndPoint = {midPoint.x + vector(0), (double) midPoint.y};
		cv::Point2d xEndPointHalf = {midPoint.x + vector(0) * 0.5, (double) midPoint.y};
		distanceX = (int) pixelsToCentimeters(vector(0));
		distanceY = -(int) pixelsToCentimeters(vector(1));
		cv::setTrackbarPos("Distance X", windowName, distanceX + MAX_DISTANCE);
		cv::setTrackbarPos("Distance Y", windowName, distanceY + MAX_DISTANCE);

		cv::line(unitCircle, midPoint, xEndPoint, cv::Scalar(0, 255, 0), 5);
		cv::line(unitCircle, xEndPoint, endPoint, cv::Scalar(0, 255, 0), 5);
		cv::putText(unitCircle, std::to_string(distanceX) + " cm", xEndPointHalf, cv::FONT_HERSHEY_PLAIN, 2,
					cv::Scalar(0, 0, 0));
		cv::putText(unitCircle, "Y: " + std::to_string(distanceY) + " cm", cv::Point{20, 80}, cv::FONT_HERSHEY_PLAIN, 2,
					cv::Scalar(0, 0, 0));

		cv::imshow(windowName, unitCircle);
		pressedKey = cv::waitKey(1);
		if (pressedKey == (int) ('q')) {
			break;
		}
	}
}