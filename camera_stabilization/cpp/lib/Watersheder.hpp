//
// Created by brucknem on 09.02.21.
//

#ifndef CAMERASTABILIZATION_WATERSHEDER_HPP
#define CAMERASTABILIZATION_WATERSHEDER_HPP

#include <utility>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include "boost/filesystem.hpp"

namespace providentia {
	namespace calibration {

		/**
		 * Wrapper for the watershed algorithm to ease the segmentation of objects.
		 */
		class Watersheder {
		private:
			/**
			 * The input and filenames.
			 */
			std::string inputFilename;

			/**
			 * The name of the main window where drawing the watershedMarkers is performed.
			 */
			std::string mainWindowName = "Watersheder";

			/**
			 * The name of the window where the segmentation is displayed.
			 */
			std::string resultWindowName = "Watersheder Result";

			/**
			 * The image buffers.
			 */
			cv::Mat image, imageGray, drawnMarkers, watershedMask, watershedMarkers;

			/**
			 * The point that holds the mouse position used to create the watershedMarkers.
			 */
			cv::Point drawingPointBuffer{-1, -1};
			/**
			 * The top left corner of the region of interest.
			 */
			cv::Point topLeftCorner{0, 0};

			/**
			 * The name of the trackbar for the top left corner x position.
			 */
			std::string posXName = "Pos X:";
			/**
			 * The name of the trackbar for the top left corner y position.
			 */
			std::string posYName = "Pos Y:";

			/**
			 * The point that holds the mouse position used to zoom in and out of the image.
			 */
			cv::Point hoverPoint{0, 0};

			/**
			 * The size of the region of interest.
			 */
			cv::Point size{0, 0};

			/**
			 * The name of the trackbar for the width.
			 */
			std::string widthName = "Width:";
			/**
			 * The name of the trackbar for the height.
			 */
			std::string heightName = "Height:";

			/**
			 * The drawing line thickness.
			 */
			int thickness = 2;
			/**
			 * The name of the trackbar for the height.
			 */
			std::string thicknessName = "Thickness:";

			/**
			 * Flag indicating whether deletion mode is on.
			 */
			bool isDeleteModeOn = false;

			/**
			 * @get The number of pixels in x direction.
			 */
			int getMaxX() const;

			/**
			 * @get The number of pixels in y direction.
			 */
			int getMaxY() const;

			/**
			 * @get The region of interest used to zoom into the image.
			 */
			cv::Rect getRoi() const;

			/**
			 * Creates an output filename based on the input filename.
			 */
			std::string createOutputFilename() const;

			/**
			 * Callback for mouse events.
			 * @param event The event code.
			 * @param x Mouse X position.
			 * @param y Mouse Y position.
			 * @param flags Additional flags.
			 * @param _watersheder The watersheder object used to apply the performed mouse action.
			 */
			static void onMouse(int event, int x, int y, int flags, void *_watersheder);

			/**
			 * Zooms in and out of the image base on the mouse position.
			 */
			void zoom(char c);

			/**
			 * Performs basic commands like closing and saving.
			 */
			void basicCommands(char c);

			/**
			 * Performs the algorithm.
			 */
			void performAlgorithm(char c);

			/**
			 * Sets the trackbar values programatically.
			 */
			void setTrackbarValues();

			/**
			 * Finds the marker contours in the drawn markers.
			 * @return The number of found marker contours.
			 */
			int findMarkerContours();

			/**
			 * Generates some random colors for the marker areas.
			 * @param amount The number of colors to generate.
			 */
			static std::vector<cv::Vec3b> generateRandomColors(int amount);

			/**
			 * Creates the color coded watershed mask.
			 * @param regions The number of found distinct regions.
			 */
			void createWatershedMask(int regions);

		protected:

			/**
			 * The algorithm implementation.
			 */
			bool algorithm();

		public:
			/**
			 * @constructor.
			 */
			explicit Watersheder(const std::string &_filename);

			/**
			 * Main loop.
			 */
			void run();

			/**
			 * Renders the image and overlays the watershedMarkers.
			 */
			cv::Mat draw() const;
		};
	}
}
#endif //CAMERASTABILIZATION_WATERSHEDER_HPP
