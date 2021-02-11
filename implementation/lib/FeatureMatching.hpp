//
// Created by brucknem on 12.01.21.
//

#ifndef CAMERASTABILIZATION_FEATUREMATCHING_HPP
#define CAMERASTABILIZATION_FEATUREMATCHING_HPP

#include "opencv2/opencv.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "FeatureDetection.hpp"
#include "TimeMeasurable.hpp"

namespace providentia {
	namespace features {

		/**
		 * Base class for all feature matchers.
		 */
		class FeatureMatcherBase : public providentia::utils::TimeMeasurable {
		private:

			/**
			 * Filters the good matches by using the fundamental matrix.
			 */
			void filterUsingFundamentalMatrix();

		protected:

			/**
			 * The current frame feature detector applied in the main loop.
			 */
			std::shared_ptr<providentia::features::FeatureDetectorBase> frameDetector, referenceFrameDetector;

			/**
			 * Flag if the fundamental matrix should be used.
			 */
			bool shouldUseFundamentalMatrix = true;

			/**
			 * Fundamental matrix inlier mask.
			 */
			cv::Mat fundamentalMatrixInlierMask;

			/**
			 * The fundamental matrix between the images.
			 */
			cv::Mat fundamentalMatrix;

			/**
			 * The k nearest neighbors of features.
			 */
			cv::cuda::GpuMat knnMatchesGPU;
			std::vector<std::vector<cv::DMatch>> knnMatchesCPU;

			/**
			 * A vector of good matches of features.
			 */
			std::vector<cv::DMatch> goodMatches;

			/**
			 * A vector of matches of features filtered using the fundamental matrix.
			 */
			std::vector<cv::DMatch> fundamentalMatches;

			/**
			 * The matched points of the frame and reference frame.
			 */
			std::vector<cv::Point2f> frameMatchedPoints, referenceMatchedPoints;

			/**
			 * The ratio threshold of good matches for the Lowe's ratio toCameraSpace.
			 */
			float goodMatchRatioThreshold;

			/**
			 * The frame with the matches drawn into.
			 */
			cv::Mat drawFrame;

			/**
			 * Specific implementation of the matching algorithm.
			 */
			virtual void specificMatch() = 0;

		public:
			/**
			 * @destructor
			 */
			~FeatureMatcherBase() override = default;

			/**
			 * @constructor
			 *
			 * @param _goodMatchRatioThreshold The ratio threshold of good matches for the Lowe's ratio toCameraSpace.
			 */
			explicit FeatureMatcherBase(float _goodMatchRatioThreshold = 0.75f);

			/**
			 * @get
			 */
			const std::vector<cv::DMatch> &getGoodMatches() const;

			/**
			 * @get
			 */
			const std::vector<cv::Point2f> &getReferenceMatchedPoints();

			/**
			 * @get
			 */
			const std::vector<cv::Point2f> &getFrameMatchedPoints() const;

			/**
			 * @set Flag for the matcher to filter the matches using the fundamental matrix.
			 */
			void setShouldUseFundamentalMatrix(bool shouldUseFundamentalMatrix);

			/**
			 * Matches the detected features of two frames.
			 *
			 * @param frameFeatureDetector The feature detector of the frame.
			 * @param referenceFeatureDetector  The feature detector of the reference frame.
			 */
			void match(const std::shared_ptr<providentia::features::FeatureDetectorBase> &_frameFeatureDetector,
					   const std::shared_ptr<providentia::features::FeatureDetectorBase> &_referenceFeatureDetector);

			/**
			 * Horizontally stacks the frames and draws the matched features.
			 */
			cv::Mat draw();

		};

		/**
		 * Brute force feature matching on GPU.
		 */
		class BruteForceFeatureMatcher : public FeatureMatcherBase {
		private:
			/**
			 * The CUDA GPU stream used during matching.
			 */
			cv::cuda::Stream stream;

			/**
			 * The Brute Force matching algorithm.
			 */
			cv::Ptr<cv::cuda::DescriptorMatcher> matcher;

		protected:
			/**
			 * @copydoc
			 */
			void specificMatch() override;

		public:
			/**
			 * @destructor
			 */
			~BruteForceFeatureMatcher() override = default;

			/**
			 * @constructor
			 *
			 * @param norm The norm used to compare the features.
			 * @param _goodMatchRatioThreshold The ratio threshold of good matches for the Lowe's ratio toCameraSpace.
			 */
			explicit BruteForceFeatureMatcher(cv::NormTypes norm, float _goodMatchRatioThreshold = 0.75f);

		};

		/**
		 * CPU implementation of the FLANN feature matcher.
		 */
		class FlannFeatureMatcher : public FeatureMatcherBase {
		private:

			/**
			 * The CPU FLANN matching algorithm.
			 */
			cv::Ptr<cv::FlannBasedMatcher> matcher;

		protected:
			/**
			 * @copydoc
			 */
			void specificMatch() override;

		public:

			/**
			 * @destructor
			 */
			~FlannFeatureMatcher() override = default;

			/**
			 * @constructor
			 *
			 * @param binaryDescriptors Flag if binary desccriptors are used.
			 * @param _goodMatchRatioThreshold The ratio threshold of good matches for the Lowe's ratio toCameraSpace.
			 */
			explicit FlannFeatureMatcher(bool binaryDescriptors = false,
										 float _goodMatchRatioThreshold = 0.75f);

			/**
			 * @constructor
			 *
			 * @params The FLANN parameters.
			 * @param _goodMatchRatioThreshold The ratio threshold of good matches for the Lowe's ratio toCameraSpace.
			 */
			explicit FlannFeatureMatcher(cv::flann::IndexParams *params, float _goodMatchRatioThreshold = 0.75f);

		};
	}
}

#endif //CAMERASTABILIZATION_FEATUREMATCHING_HPP
