//
// Created by brucknem on 13.01.21.
//

#include "opencv2/opencv.hpp"
#include "Commons.hpp"
#include "boost/program_options.hpp"

#pragma region Helpers

std::string providentia::runnables::getDefaultVideoFile() {
    std::string basePath = "/mnt/local_data/providentia/test_recordings/videos/";
    std::string filename = "s40_n_far_image_raw";
    std::string suffix = ".mp4";
    std::stringstream fullPath;
    fullPath << basePath << filename << suffix;
    return fullPath.str();
}

cv::VideoCapture providentia::runnables::openVideoCapture(const std::string &videoFileName) {
    cv::VideoCapture cap(videoFileName);

    if (!cap.isOpened()) // if not success, exit program
    {
        throw std::invalid_argument("Cannot open " + videoFileName);
    }

    return cap;
}


std::string providentia::runnables::durationInfo(const std::string &message, long milliseconds) {
    std::stringstream ss;
    ss << message << " - Duration: " << milliseconds << "ms - FPS: " << 1000. / milliseconds;
    return ss.str();
}

void providentia::runnables::addText(const cv::Mat &frame, const std::string &text, int x, int y) {
    cv::putText(frame, text, cv::Point(x, y),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 0), 2, cv::FONT_HERSHEY_SIMPLEX);
}

cv::Mat providentia::runnables::pad(const cv::Mat &frame, int padding) {
    return cv::Mat(frame,
                   cv::Rect(padding, padding, frame.cols - 2 * padding, frame.rows - 2 * padding));
}


void providentia::runnables::addRuntimeToFrame(const cv::Mat &_frame, const std::string &message,
                                               long milliseconds, int x,
                                               int y) {
    addText(_frame, durationInfo(message, milliseconds), x, y);
}

#pragma endregion Helpers

#pragma region RunnablesCommons

providentia::runnables::BaseSetup::BaseSetup(std::string _videoFileName,
                                             std::string _windowName,
                                             double _calculationScaleFactor, double _renderingScaleFactor)
        : videoFileName(std::move(_videoFileName)), calculationScaleFactor(_calculationScaleFactor),
          renderingScaleFactor(_renderingScaleFactor),
          windowName(std::move(_windowName)) {
    init();
}

providentia::runnables::BaseSetup::BaseSetup(int argc, const char **argv) {
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

void providentia::runnables::BaseSetup::init() {
    this->capture = providentia::runnables::openVideoCapture(videoFileName);
    this->renderingScaleFactor /= this->calculationScaleFactor;
    cv::namedWindow(this->windowName, cv::WINDOW_AUTOSIZE);
}

providentia::runnables::BaseSetup::~BaseSetup() {
    capture.release();
    cv::destroyAllWindows();
}

void providentia::runnables::BaseSetup::addRuntimeToFinalFrame(const std::string &text, long milliseconds, int x,
                                                               int y) {
    addTextToFinalFrame(durationInfo(text, milliseconds), x, y);
}

void providentia::runnables::BaseSetup::addTextToFinalFrame(const std::string &text, int x, int y) {
    addText(finalFrame, text, x, y);
}

void providentia::runnables::BaseSetup::mainLoop() {
    while (true) {
        capture >> frameCPU;
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
            addRuntimeToFinalFrame("Algorithms total", totalAlgorithmsDuration, 5, finalFrame.size().height - 30);
        }

        specificAddMessages();

        cv::imshow(windowName, finalFrame);

        if (cv::waitKey(1) == (int) ('q')) {
            break;
        }
    }
}

void providentia::runnables::BaseSetup::specificAddMessages() {
    // Empty stub for optional override.
}

void providentia::runnables::BaseSetup::setCalculationScaleFactor(double calculationScaleFactor) {
    BaseSetup::calculationScaleFactor = calculationScaleFactor;
}

void providentia::runnables::BaseSetup::setRenderingScaleFactor(double renderingScaleFactor) {
    BaseSetup::renderingScaleFactor = renderingScaleFactor;
}

void providentia::runnables::BaseSetup::setCapture(const std::string &file) {
    if (BaseSetup::capture.isOpened()) {
        BaseSetup::capture.release();
    }
    BaseSetup::capture = openVideoCapture(file);
}

#pragma endregion RunnablesCommons
