//
// Created by brucknem on 13.01.21.
//

#ifndef CAMERASTABILIZATION_BACKGROUNDSEGMENTATION_HPP
#define CAMERASTABILIZATION_BACKGROUNDSEGMENTATION_HPP

#include <opencv2/cudabgsegm.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/cudafilters.hpp"
#include "TimeMeasurable.hpp"

namespace providentia {
	namespace segmentation {
		/**
		 * Base class for the background segmentation wrappers.
		 */
		class BackgroundSegmentionBase : public providentia::utils::TimeMeasurable {
		private:

			/**
			 * An internal used mask consisting of on 255 values used for
			 * calculating the background mask as the difference of 255 - foreground mask.
			 */
			cv::cuda::GpuMat all255Mask;

		protected:
			/**
			 * The size used for background calculation.
			 */
			cv::Size calculationSize;
			/**
			 * A CUDA stream used for async calculations.
			 */
			cv::cuda::Stream stream;

			/**
			 * The frame used during calculations.
			 */
			cv::cuda::GpuMat calculationFrame;

			/**
			 * The fore- and background masks.
			 */
			cv::cuda::GpuMat foregroundMask, backgroundMask;

			/**
			 * Some additional filters used during post processing.
			 */
			std::vector<cv::Ptr<cv::cuda::Filter>> filters;

			/**
			 * Subclass specifc algorithm implementation.
			 */
			virtual void specificApply() = 0;

			/**
			 * Additional postprocessing steps on the raw background segmentation result of the algorithm.
			 */
			void postProcess();

			/**
			 * @constructor
			 */
			explicit BackgroundSegmentionBase(cv::Size calculationSize = cv::Size());

			/**
			 * Add optional filters to the postprocessing step.
			 */
			virtual void addFilters();

		public:

			/**
			 * @destructor
			 */
			~BackgroundSegmentionBase() override = default;

			/**
			 * Appends the given frame to the internal history of writeFrames and calculates the background segmentation.
			 *
			 * @param frame The new frame to apply.
			 */
			virtual void segment(const cv::cuda::GpuMat &frame);

			/**
			 * Clears the masks to all background.
			 */
			void clearMasks(const cv::Size &size = cv::Size());

			/**
			 * @get
			 *
			 * @return The background mask.
			 */
			const cv::cuda::GpuMat &getBackgroundMask(const cv::Size &size = cv::Size());

			/**
			 * @get
			 *
			 * @return The foreground mask.
			 */
			const cv::cuda::GpuMat &getForegroundMask(const cv::Size &size = cv::Size());

			/**
			 * Draws the foreground and background masks aside.
			 */
			cv::Mat draw() const;
		};

		/**
		 * Wrapper for the MOG2 background segmentation algorithm.
		 */
		class MOG2BackgroundSegmention : public BackgroundSegmentionBase {
		private:

			/**
			 * The algorithm implementation.
			 */
			cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> algorithm;

		protected:

			/**
			 * @copydoc
			 */
			void specificApply() override;

			/**
			 * @copydoc
			 */
			void addFilters() override;

		public:

			/**
			 * @constructor
			 *
			 * @param calculationSize The scaling size of the frame during calculation.
			 * @param history Length of the history.
			 * @param varThreshold Threshold on the squared Mahalanobis distance between the expectedPixel and the model
			 * to decide whether a expectedPixel is well described by the background model. This parameter does not
			 * affect the background update.
			 * @param detectShadows If true, the algorithm will detect shadows and mark them. It decreases the
			 * speed a bit, so if you do not need this feature, set the parameter to false.
			 */
			explicit MOG2BackgroundSegmention(cv::Size calculationSize = cv::Size(), int history = 500,
											 double varThreshold = 16,
											 bool detectShadows = false);

			/**
			 * @desctructor
			 */
			~MOG2BackgroundSegmention() override = default;
		};
	}
}

#endif //CAMERASTABILIZATION_BACKGROUNDSEGMENTATION_HPP
