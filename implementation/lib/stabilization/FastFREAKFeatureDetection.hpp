//
// Created by brucknem on 11.04.21.
//

#ifndef CAMERASTABILIZATION_FASTFREAKFEATUREDETECTION_HPP
#define CAMERASTABILIZATION_FASTFREAKFEATUREDETECTION_HPP

#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include <opencv2/cudafeatures2d.hpp>

#include "TimeMeasurable.hpp"
#include "FeatureDetectionBase.hpp"

namespace providentia {
	namespace stabilization {
		namespace detection {

/**
 * Wrapper for the CUDA SURF feature detector.
 */
			class FastFREAKFeatureDetection : public providentia::stabilization::detection::FeatureDetectionBase {
			private:
				/**
				 * The CUDA FastFeature detector used to detect keypoints and descriptors.
				 */
				cv::Ptr<cv::cuda::FastFeatureDetector> detector;
				cv::Ptr<cv::xfeatures2d::FREAK> descriptor;

			protected:
				/**
				 * @copydoc
				 */
				void specificDetect() override;

			public:

				/**
				 * @constructor
				 *
				 * @ref opencv2/cudafeatures2d.hpp -> cv::cuda::FastFeatures::create
				 */
				explicit FastFREAKFeatureDetection(int threshold = 40,
												   bool nonmaxSuppression = true,
												   cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16,
												   int maxNPoints = 500000,
												   bool orientationNormalized = true,
												   bool scaleNormalized = true,
												   float patternScale = 22.0f,
												   int nOctaves = 4,
												   const std::vector<int> &selectedPairs = std::vector<int>());

				/**
				 * @destructor
				 */
				~FastFREAKFeatureDetection() override = default;
			};

		}
	}
}
#endif //CAMERASTABILIZATION_FASTFREAKFEATUREDETECTION_HPP
