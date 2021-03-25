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

std::string providentia::evaluation::getDefaultVideoFile() {
	std::string basePath = "/mnt/local_data/providentia/test_recordings/videos/";
	std::string filename = "s40_n_far_image_raw";
	std::string suffix = ".mp4";
	std::stringstream fullPath;
	fullPath << basePath << filename << suffix;
	return fullPath.str();
}

cv::VideoCapture providentia::evaluation::openVideoCapture(const std::string &videoFileName) {
	cv::VideoCapture cap(videoFileName);

	if (!cap.isOpened()) // if not success, exit program
	{
		throw std::invalid_argument("Cannot open " + videoFileName);
	}

	return cap;
}

std::string providentia::evaluation::durationInfo(const std::string &message, long milliseconds) {
	std::stringstream ss;
	ss << message << " - Duration: " << milliseconds << "ms - FPS: " << 1000. / milliseconds;
	return ss.str();
}

cv::Mat providentia::evaluation::addText(const cv::Mat &frame, const std::string &text, double fontSize, int x, int y) {
	cv::putText(frame, text, cv::Point(x, y + 20 * fontSize),
				cv::FONT_HERSHEY_COMPLEX_SMALL, fontSize, cv::Scalar(255, 255, 0), fontSize, cv::FONT_HERSHEY_SIMPLEX);
	return frame;
}

cv::Mat providentia::evaluation::pad(const cv::Mat &frame, int padding) {
	return cv::Mat(frame,
				   cv::Rect(padding, padding, frame.cols - 2 * padding, frame.rows - 2 * padding));
}

cv::Mat providentia::evaluation::addRuntimeToFrame(const cv::Mat &_frame, const std::string &message,
												   long milliseconds, double fontSize, int x, int y) {
	return addText(_frame, durationInfo(message, milliseconds), fontSize, x, y);
}

double providentia::evaluation::getRandom01() {
	return static_cast<double>(rand() / static_cast<double>(RAND_MAX));
}

template<typename T>
std::vector<cv::Mat> providentia::evaluation::concatenate(const std::initializer_list<T> &frames, cv::Size size) {
	std::vector<cv::Mat> concatenated;
	cv::Mat resizeBuffer;

	cv::Size _size;
	for (const auto &frame : frames) {
		concatenated.emplace_back(cv::Mat(frame));
		if (_size.empty()) {
			if (size.empty()) {
				_size = concatenated[concatenated.size() - 1].size();
			} else {
				_size = size;
			}
		}
		if (!size.empty()) {
			cv::resize(concatenated[concatenated.size() - 1], concatenated[concatenated.size() - 1], _size);
		}
	}

	return concatenated;
}

template<typename T>
cv::Mat providentia::evaluation::vconcat(const std::initializer_list<T> &frames, cv::Size size) {
	cv::Mat result;
	cv::vconcat(concatenate<T>(frames, size), result);
	return result;
}

template cv::Mat providentia::evaluation::vconcat(const std::initializer_list<cv::Mat> &, cv::Size size);

template cv::Mat providentia::evaluation::vconcat(const std::initializer_list<cv::cuda::GpuMat> &, cv::Size size);

template<typename T>
cv::Mat providentia::evaluation::hconcat(const std::initializer_list<T> &frames, cv::Size size) {
	cv::Mat result;
	cv::hconcat(concatenate<T>(frames, size), result);
	return result;
}

template cv::Mat providentia::evaluation::hconcat(const std::initializer_list<cv::Mat> &, cv::Size size);

template cv::Mat providentia::evaluation::hconcat(const std::initializer_list<cv::cuda::GpuMat> &, cv::Size size);

cv::Mat providentia::evaluation::MatOfSize(cv::Size size, int type) {
	return cv::Mat(cv::Mat::zeros(std::move(size), type));
}

cv::cuda::GpuMat providentia::evaluation::GpuMatOfSize(cv::Size size, int type) {
	cv::cuda::GpuMat mat;
	mat.upload(cv::Mat(cv::Mat::zeros(std::move(size), type)));
	return mat;
}

cv::cuda::GpuMat providentia::evaluation::cvtColor(cv::cuda::GpuMat frame, int colorSpace) {
	cv::cuda::GpuMat result;
	cv::cuda::cvtColor(frame, result, colorSpace);
	return result;
}

#pragma endregion Helpers

#pragma region RunnablesCommons

providentia::evaluation::ImageSetup::ImageSetup(std::string _filename,
												std::string _windowName,
												double _calculationScaleFactor, double _renderingScaleFactor)
	: filename(std::move(_filename)), calculationScaleFactor(_calculationScaleFactor),
	  renderingScaleFactor(_renderingScaleFactor),
	  windowName(std::move(_windowName)) {
	frameCPU = cv::imread(filename);
	init();
}

void providentia::evaluation::ImageSetup::init() {
	renderingScaleFactor /= calculationScaleFactor;
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
	cv::moveWindow(windowName, 50, 10);
	srand(static_cast <unsigned> (time(0)));
}

void providentia::evaluation::ImageSetup::setWindowMode(int flags) {
	cv::destroyWindow(windowName);
	cv::namedWindow(windowName, flags);
}

void providentia::evaluation::ImageSetup::addRuntimeToFinalFrame(const std::string &text, long milliseconds, int x,
																 int y) {
	addTextToFinalFrame(durationInfo(text, milliseconds), x, y);
}

void providentia::evaluation::ImageSetup::addTextToFinalFrame(const std::string &text, int x, int y) {
	addText(finalFrame, text, 1, x, y);
}

void providentia::evaluation::ImageSetup::getNextFrame() {
//pass
}

void providentia::evaluation::ImageSetup::mainLoop() {
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
//		addRuntimeToFinalFrame("Frame " + std::to_string(frameNumber), getTotalMilliseconds(), 5, finalFrame.rows - 20);

		cv::imshow(windowName, finalFrame);
		if (!outputFolder.empty()) {
			finalFrame *= 255;
			finalFrame.convertTo(finalFrame, CV_8UC4);
			std::stringstream frameId;
			frameId.fill('0');
			frameId.width(5);
			frameId << frameNumber;
			boost::filesystem::path outFile = outputFolder / ("frame_" + frameId.str() + ".jpg");
			cv::imwrite(outFile.string(), finalFrame, {cv::IMWRITE_PNG_COMPRESSION, 9});
		}
		pressedKey = cv::waitKey(1);
		if (pressedKey == (int) ('q')) {
			break;
		}

		frameNumber++;
	}
}

void providentia::evaluation::ImageSetup::specificAddMessages() {
	// Empty stub for optional override.
}

void providentia::evaluation::ImageSetup::setCalculationScaleFactor(double _calculationScaleFactor) {
	calculationScaleFactor = _calculationScaleFactor;
}

void providentia::evaluation::ImageSetup::setRenderingScaleFactor(double _renderingScaleFactor) {
	renderingScaleFactor = _renderingScaleFactor;
}

void providentia::evaluation::ImageSetup::setOutputFolder(const std::string &_outputFolder) {
	outputFolder = _outputFolder;
	if (!boost::filesystem::is_directory(outputFolder)) {
		boost::filesystem::create_directories(outputFolder);
	}
}

#pragma endregion RunnablesCommons

void providentia::evaluation::VideoSetup::setCapture(const std::string &file) {
	if (capture.isOpened()) {
		capture.release();
	}
	capture = openVideoCapture(file);
}

providentia::evaluation::VideoSetup::VideoSetup(std::string _videoFileName, std::string _windowName,
												double _calculationScaleFactor, double _renderingScaleFactor)
	: ImageSetup(std::move(_videoFileName), std::move(_windowName), _calculationScaleFactor, _renderingScaleFactor) {
	capture = providentia::evaluation::openVideoCapture(filename);
	init();
}

providentia::evaluation::VideoSetup::~VideoSetup() {
	capture.release();
	cv::destroyAllWindows();
}

void providentia::evaluation::VideoSetup::getNextFrame() {
	capture >> frameCPU;
}