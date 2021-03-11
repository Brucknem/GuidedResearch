//
// Created by brucknem on 13.01.21.
//

#include "opencv2/opencv.hpp"
#include "Commons.hpp"

#include <utility>
#include "boost/program_options.hpp"
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#pragma region Helpers

std::string providentia::runnable::getDefaultVideoFile() {
	std::string basePath = "/mnt/local_data/providentia/test_recordings/videos/";
	std::string filename = "s40_n_far_image_raw";
	std::string suffix = ".mp4";
	std::stringstream fullPath;
	fullPath << basePath << filename << suffix;
	return fullPath.str();
}

cv::VideoCapture providentia::runnable::openVideoCapture(const std::string &videoFileName) {
	cv::VideoCapture cap(videoFileName);

	if (!cap.isOpened()) // if not success, exit program
	{
		throw std::invalid_argument("Cannot open " + videoFileName);
	}

	return cap;
}

std::string providentia::runnable::durationInfo(const std::string &message, long milliseconds) {
	std::stringstream ss;
	ss << message << " - Duration: " << milliseconds << "ms - FPS: " << 1000. / milliseconds;
	return ss.str();
}

void providentia::runnable::addText(const cv::Mat &frame, const std::string &text, int x, int y) {
	cv::putText(frame, text, cv::Point(x, y),
				cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 0), 1, cv::FONT_HERSHEY_SIMPLEX);
}

cv::Mat providentia::runnable::pad(const cv::Mat &frame, int padding) {
	return cv::Mat(frame,
				   cv::Rect(padding, padding, frame.cols - 2 * padding, frame.rows - 2 * padding));
}

void providentia::runnable::addRuntimeToFrame(const cv::Mat &_frame, const std::string &message,
											  long milliseconds, int x,
											  int y) {
	addText(_frame, durationInfo(message, milliseconds), x, y);
}

double providentia::runnable::getRandom01() {
	return static_cast<double>((rand()) / static_cast<double>(RAND_MAX));
}

#pragma endregion Helpers

#pragma region RunnablesCommons

providentia::runnable::ImageSetup::ImageSetup(std::string _filename,
											  std::string _windowName,
											  double _calculationScaleFactor, double _renderingScaleFactor)
	: filename(std::move(_filename)), calculationScaleFactor(_calculationScaleFactor),
	  renderingScaleFactor(_renderingScaleFactor),
	  windowName(std::move(_windowName)) {
	frameCPU = cv::imread(filename);
	init();
}

void providentia::runnable::ImageSetup::init() {
	renderingScaleFactor /= calculationScaleFactor;
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
	srand(static_cast <unsigned> (time(0)));
}

void providentia::runnable::ImageSetup::setWindowMode(int flags) {
	cv::destroyWindow(windowName);
	cv::namedWindow(windowName, flags);
}

void providentia::runnable::ImageSetup::addRuntimeToFinalFrame(const std::string &text, long milliseconds, int x,
															   int y) {
	addTextToFinalFrame(durationInfo(text, milliseconds), x, y);
}

void providentia::runnable::ImageSetup::addTextToFinalFrame(const std::string &text, int x, int y) {
	addText(finalFrame, text, x, y);
}

void providentia::runnable::ImageSetup::getNextFrame() {
//pass
}

void providentia::runnable::ImageSetup::mainLoop() {
	while (true) {
		clear();
		getNextFrame();
		if (frameCPU.empty()) {
			break;
		}
		finalFrame = cv::Mat();
		cv::resize(frameCPU, frameCPU, cv::Size(), calculationScaleFactor, calculationScaleFactor);
		frameGPU.upload(frameCPU);

		totalAlgorithmsDuration = 0;
		specificMainLoop();

		cv::resize(finalFrame, finalFrame, cv::Size(), renderingScaleFactor, renderingScaleFactor);

		if (totalAlgorithmsDuration > 0) {
			addRuntimeToFinalFrame("Algorithms total", totalAlgorithmsDuration, 5, finalFrame.rows - 40);
		}

		specificAddMessages();

		addTimestamp("End");
		addRuntimeToFinalFrame("Frame " + std::to_string(frameNumber), getTotalMilliseconds(), 5, finalFrame.rows - 20);

		cv::imshow(windowName, finalFrame);
		if (!outputFolder.empty()) {
			finalFrame *= 255;
			finalFrame.convertTo(finalFrame, CV_8UC4);
			boost::filesystem::path outFile = outputFolder / ("frame_" + std::to_string(frameNumber) + ".jpg");
			cv::imwrite(outFile.string(), finalFrame, {cv::IMWRITE_PNG_COMPRESSION, 9});
		}
		pressedKey = cv::waitKey(1);
		if (pressedKey == (int) ('q')) {
			break;
		}

		frameNumber++;
	}
}

void providentia::runnable::ImageSetup::specificAddMessages() {
	// Empty stub for optional override.
}

void providentia::runnable::ImageSetup::setCalculationScaleFactor(double _calculationScaleFactor) {
	calculationScaleFactor = _calculationScaleFactor;
}

void providentia::runnable::ImageSetup::setRenderingScaleFactor(double _renderingScaleFactor) {
	renderingScaleFactor = _renderingScaleFactor;
}

void providentia::runnable::ImageSetup::setOutputFolder(const std::string &_outputFolder) {
	outputFolder = _outputFolder;
	if (!boost::filesystem::is_directory(outputFolder)) {
		boost::filesystem::create_directories(outputFolder);
	}
}

#pragma endregion RunnablesCommons

void providentia::runnable::VideoSetup::setCapture(const std::string &file) {
	if (capture.isOpened()) {
		capture.release();
	}
	capture = openVideoCapture(file);
}

providentia::runnable::VideoSetup::VideoSetup(std::string _videoFileName, std::string _windowName,
											  double _calculationScaleFactor, double _renderingScaleFactor)
	: ImageSetup(std::move(_videoFileName), std::move(_windowName), _calculationScaleFactor, _renderingScaleFactor) {
	capture = providentia::runnable::openVideoCapture(filename);
	init();
}

providentia::runnable::VideoSetup::~VideoSetup() {
	capture.release();
	cv::destroyAllWindows();
}

void providentia::runnable::VideoSetup::getNextFrame() {
	capture >> frameCPU;
}