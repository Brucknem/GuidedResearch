#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <cstdio>
#include <iostream>

using namespace cv;
using namespace std;

Mat markerMask, img, roiImage;
Point prevPt(-1, -1);

static void onMouse(int event, int x, int y, int flags, void *) {
	if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
		return;
	if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
		prevPt = Point(-1, -1);
	else if (event == EVENT_LBUTTONDOWN)
		prevPt = Point(x, y);
	else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {
		Point pt(x, y);
		if (prevPt.x < 0)
			prevPt = pt;
		line(markerMask, prevPt, pt, Scalar::all(255), 5, 8, 0);
		line(roiImage, prevPt, pt, Scalar::all(255), 5, 8, 0);
		prevPt = pt;
		imshow("image", roiImage);
	}
}

static void zoomIn(int event, int x, int y, int flags, void *) {
	if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
		return;
	if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON)) {
		Point pt(x, y);
		cv::Mat copy = img.clone();
		cv::Rect roi(prevPt, pt);
		if (roi.x < 0) {
			return;
		}
		copy = copy(roi);
		double aspect = roi.width * 1. / roi.height;
		cv::resize(copy, copy, cv::Size(800 * aspect, 800));
		imshow("roi", copy);
		setMouseCallback("roi", onMouse, 0);
		prevPt = Point(-1, -1);
	} else if (event == EVENT_LBUTTONDOWN) {
		Point pt(x, y);
		prevPt = pt;
	} else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {
		Point pt(x, y);
		if (prevPt.x < 0)
			prevPt = pt;
//		line(markerMask, prevPt, pt, Scalar::all(255), 5, 8, 0);
		cv::Mat copy = img.clone();
		rectangle(copy, prevPt, pt, Scalar::all(255), 3, 8, 0);
//		prevPt = pt;
		imshow("image", copy);
	}
}

int main(int argc, char **argv) {
	string filename = "../misc/s40_n_cam_far_calibration_test_image.png";
	Mat img0 = imread(filename, 1), imgGray;
	namedWindow("image", 1);
	img0.copyTo(img);
	cvtColor(img, markerMask, COLOR_BGR2GRAY);
	cvtColor(markerMask, imgGray, COLOR_GRAY2BGR);
	markerMask = Scalar::all(0);
	imshow("image", img);
	setMouseCallback("image", zoomIn, 0);
	for (;;) {
		char c = (char) waitKey(0);
		if (c == 27)
			break;
		if (c == 'r') {
			markerMask = Scalar::all(0);
			img0.copyTo(img);
			imshow("image", img);
		}
		if (c == 'w' || c == ' ') {
			int i, j, compCount = 0;
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			findContours(markerMask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
			if (contours.empty())
				continue;
			Mat markers(markerMask.size(), CV_32S);
			markers = Scalar::all(0);
			int idx = 0;
			for (; idx >= 0; idx = hierarchy[idx][0], compCount++)
				drawContours(markers, contours, idx, Scalar::all(compCount + 1), -1, 8, hierarchy, INT_MAX);
			if (compCount == 0)
				continue;
			vector<Vec3b> colorTab;
			for (i = 0; i < compCount; i++) {
				int b = theRNG().uniform(0, 255);
				int g = theRNG().uniform(0, 255);
				int r = theRNG().uniform(0, 255);
				colorTab.push_back(Vec3b((uchar) b, (uchar) g, (uchar) r));
			}
			double t = (double) getTickCount();
			watershed(img0, markers);
			t = (double) getTickCount() - t;
			printf("execution time = %gms\n", t * 1000. / getTickFrequency());
			Mat wshed(markers.size(), CV_8UC3);
			// paint the watershed image
			for (i = 0; i < markers.rows; i++)
				for (j = 0; j < markers.cols; j++) {
					int index = markers.at<int>(i, j);
					if (index == -1)
						wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
					else if (index <= 0 || index > compCount)
						wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
					else
						wshed.at<Vec3b>(i, j) = colorTab[index - 1];
				}
			wshed = wshed * 0.5 + imgGray * 0.5;
			imshow("watershed transform", wshed);
		}
	}
	return 0;
}