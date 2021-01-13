//
// Created by brucknem on 13.01.21.
//

#ifndef CAMERASTABILIZATION_RUNNABLESCOMMONS_HPP
#define CAMERASTABILIZATION_RUNNABLESCOMMONS_HPP

#include <stdexcept>
#include "opencv2/opencv.hpp"
#include "../library/Utils.hpp"

namespace providentia {
    namespace runnables {

        /**
         * Gets the default video file path.
         */
        std::string getDefaultVideoFile();

        /**
         * Opens the given path and loads it as a video capture.
         *
         * @param videoFileName The full file name of the video.
         * @return A video capture with the loaded video.
         * @throws std::invalid_argument if video not loaded
         */
        cv::VideoCapture openVideoCapture(const std::string &videoFileName);

        /**
         * Generates a readable string containing the message, duration and fps.
         *
         * @param message The message prefix.
         * @param milliseconds The duration in milliseconds.
         * @return A formatted string with the duration.
         */
        std::string durationInfo(const std::string &message, long milliseconds);

        /**
         * Adds text to a frame.
         *
         * @param frame The frame to add text to.
         * @param text The text to add.
         * @param x The left starting pixel location.
         * @param y The upper starting pixel location.
         */
        void addText(const cv::Mat &frame, const std::string &text, int x, int y);

        /**
         * Adds runtime info to the frame.
         *
         * @param frame The frame to add message to.
         * @param message The message to add.
         * @param milliseconds The duration in milliseconds.
         * @param x The left starting pixel location.
         * @param y The upper starting pixel location.
         */
        void addRuntimeToFrame(const cv::Mat &_frame, const std::string &message, long milliseconds, int x, int y);

        /**
         * Removes the padding pixels from the frame.
         *
         * @param frame The frame to remove the pixels from.
         * @param padding The number of pixels to remove from each side.
         * @return The padded frame.
         */
        cv::Mat pad(const cv::Mat &frame, int padding);

        /**
         * Base class for all runnable setups.
         * Wraps the main loop and field initializations.
         */
        class BaseSetup : providentia::utils::TimeMeasurable {
        protected:
            /**
             * The current CPU frame.
             */
            cv::Mat frameCPU;
            /**
             * The final CPU frame after processing. This will be rendered at the end.
             */
            cv::Mat finalFrame;

            /**
             * The current GPU frame.
             */
            cv::cuda::GpuMat frameGPU;

            /**
             * A CPU frame buffer for additional calculations.
             */
            cv::Mat bufferCPU;

            /**
             * A GPU frame buffer for additional calculations.
             */
            cv::cuda::GpuMat bufferGPU;

            /**
             * The file name of the video.
             */
            std::string videoFileName;

            /**
             * The video capture device.
             */
            cv::VideoCapture capture;

            /**
             * The title name of the window.
             */
            std::string windowName;

            /**
             * A scaling factor applied to the frame before calculation.
             */
            double calculationScaleFactor = 1;

            /**
             * A scaling factor applied to the final frame before rendering.
             */
            double renderingScaleFactor = 0.5;

            /**
             * The total duration of the algorithms.
             */
            long totalAlgorithmsDuration = 0;

            /**
             * The subclass specific main loop. All calculation is done here. <br>
             * Has to finally set the finalFrame which will then be rendered.
             */
            virtual void specificMainLoop() = 0;

            /**
             * Add some optional specific messages to the final frame after the main processing.
             */
            virtual void specificAddMessages();

            /**
             * Adds some text to the final frame.
             *
             * @param text The text to add.
             * @param x The left starting pixel location.
             * @param y The upper starting pixel location.
             */
            void addTextToFinalFrame(const std::string &text, int x, int y);

            /**
             * Adds some text to the final frame.
             *
             * @param text The text to add.
             * @param milliseconds The runtime in milliseconds.
             * @param x The left starting pixel location.
             * @param y The upper starting pixel location.
             */
            void addRuntimeToFinalFrame(const std::string &text, long milliseconds, int x, int y);

            /**
             * Initializer for the fields that are derived from the arguments.
             */
            void init();

        public:

            /**
             * @constructor Parses the given command line arguments and sets the field.
             *
             * @param argc Number of command line arguments.
             * @param argv Command line arguments.
             */
            explicit BaseSetup(int argc, char const *argv[]);

            /**
             * @constructor Directly sets the fields.
             *
             * @param _videoFileName The video file name.
             * @param _windowName The name of the rendering window.
             * @param _calculationScaleFactor The scale factor of frame during calculation.
             * @param _renderingScaleFactor The scale factor of final frame during rendering.
             */
            explicit BaseSetup(std::string _videoFileName = getDefaultVideoFile(),
                               std::string _windowName = "Camera Stabilization",
                               double _calculationScaleFactor = 1,
                               double _renderingScaleFactor = 0.5);

            /**
             * @destructor
             */
            ~BaseSetup();

            /**
             * The main loop of the setup. <br>
             * Retrieves the next frame, does the processing, renders the results and measures the total duration.
             */
            void mainLoop();
        };

    }
}


#endif //CAMERASTABILIZATION_RUNNABLESCOMMONS_HPP
