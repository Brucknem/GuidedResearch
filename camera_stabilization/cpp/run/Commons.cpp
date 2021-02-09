//
// Created by brucknem on 13.01.21.
//

#include "opencv2/opencv.hpp"
#include "Commons.hpp"
#include "boost/program_options.hpp"

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
				cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 0), 2, cv::FONT_HERSHEY_SIMPLEX);
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

providentia::runnable::BaseSetup::BaseSetup(std::string _videoFileName,
											std::string _windowName,
											double _calculationScaleFactor, double _renderingScaleFactor)
		: videoFileName(std::move(_videoFileName)), calculationScaleFactor(_calculationScaleFactor),
		  renderingScaleFactor(_renderingScaleFactor),
		  windowName(std::move(_windowName)) {
	init();
}

providentia::runnable::BaseSetup::BaseSetup(int argc, const char **argv) : providentia::utils::TimeMeasurable("Total",
																											  1) {
	boost::program_options::options_description description("Allowed options");
	description.add_options()
			("help", "Produce help message.")
			("videoFileName",
			 boost::program_options::value<std::string>(&videoFileName)->default_value(getDefaultVideoFile()),
			 "The path to the video file.")
			("windowName",
			 boost::program_options::value<std::string>(&windowName)->default_value("Camera Stabilization"),
			 "The window name.")
			("calcScale", boost::program_options::value<double>(&calculationScaleFactor)->default_value(1.0),
			 "The scale factor of frame during calculation.")
			("renderScale", boost::program_options::value<double>(&renderingScaleFactor)->default_value(0.5),
			 "The scale factor of final frame during rendering.");

	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::parse_command_line(argc, argv, description), vm);
	boost::program_options::notify(vm);

	if (vm.count("help")) {
		std::cout << description << "\n";
		exit(0);
	}
	init();
}

void providentia::runnable::BaseSetup::init() {
	this->capture = providentia::runnable::openVideoCapture(videoFileName);
	this->renderingScaleFactor /= this->calculationScaleFactor;
	cv::namedWindow(this->windowName, cv::WINDOW_AUTOSIZE);
	srand(static_cast <unsigned> (time(0)));
}

void providentia::runnable::BaseSetup::setWindowMode(int flags) {
	cv::destroyWindow(windowName);
	cv::namedWindow(windowName, flags);
}

providentia::runnable::BaseSetup::~BaseSetup() {
	capture.release();
	cv::destroyAllWindows();
}

void providentia::runnable::BaseSetup::addRuntimeToFinalFrame(const std::string &text, long milliseconds, int x,
															  int y) {
	addTextToFinalFrame(durationInfo(text, milliseconds), x, y);
}

void providentia::runnable::BaseSetup::addTextToFinalFrame(const std::string &text, int x, int y) {
	addText(finalFrame, text, x, y);
}

void providentia::runnable::BaseSetup::getNextFrame() {
	capture >> frameCPU;
}

void providentia::runnable::BaseSetup::mainLoop() {
	while (true) {
		clear();
		getNextFrame();
		if (frameCPU.empty()) {
			break;
		}
		finalFrame = cv::Mat();
		cv::Mat originalFrame = frameCPU.clone();
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
		addRuntimeToFinalFrame("Frame", getTotalMilliseconds(), 5, finalFrame.rows - 20);

		cv::imshow(windowName, finalFrame);

		if (cv::waitKey(1) == (int) ('q')) {
			break;
		}
	}
}

void providentia::runnable::BaseSetup::specificAddMessages() {
	// Empty stub for optional override.
}

void providentia::runnable::BaseSetup::setCalculationScaleFactor(double _calculationScaleFactor) {
	calculationScaleFactor = _calculationScaleFactor;
}

void providentia::runnable::BaseSetup::setRenderingScaleFactor(double _renderingScaleFactor) {
	renderingScaleFactor = _renderingScaleFactor;
}

void providentia::runnable::BaseSetup::setCapture(const std::string &file) {
	if (BaseSetup::capture.isOpened()) {
		BaseSetup::capture.release();
	}
	BaseSetup::capture = openVideoCapture(file);
}

#pragma endregion RunnablesCommons
