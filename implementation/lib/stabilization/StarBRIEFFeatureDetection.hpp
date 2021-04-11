//
// Created by brucknem on 11.04.21.
//

#ifndef CAMERASTABILIZATION_STARBRIEFFEATUREDETECTION_HPP
#define CAMERASTABILIZATION_STARBRIEFFEATUREDETECTION_HPP

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
			class StarBRIEFFeatureDetection : public providentia::stabilization::detection::FeatureDetectionBase {
			private:
				/**
				 * The CUDA FastFeature detector used to detect keypoints and descriptors.
				 */
				cv::Ptr<cv::xfeatures2d::StarDetector> detector;
				cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> descriptor;

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
				explicit StarBRIEFFeatureDetection(int maxSize = 45, int responseThreshold = 30,
												   int lineThresholdProjected = 10,
												   int lineThresholdBinarized = 8,
												   int suppressNonmaxSize = 5,
												   int bytes = 64, bool useOrientation = false);

				/**
				 * @destructor
				 */
				~StarBRIEFFeatureDetection() override = default;

			};

		}
	}
}
#endif //CAMERASTABILIZATION_STARBRIEFFEATUREDETECTION_HPP
