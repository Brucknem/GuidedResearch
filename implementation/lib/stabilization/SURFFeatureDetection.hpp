//
// Created by brucknem on 11.04.21.
//

#ifndef CAMERASTABILIZATION_SURFFEATUREDETECTION_HPP
#define CAMERASTABILIZATION_SURFFEATUREDETECTION_HPP

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
			class SURFFeatureDetection : public providentia::stabilization::detection::FeatureDetectionBase {
			private:
				/**
				 * The CUDA SURF detector used to detect keypoints and descriptors.
				 */
				cv::Ptr<cv::cuda::SURF_CUDA> detector;

			protected:
				/**
				 * @copydoc
				 */
				void specificDetect() override;

			public:

				/**
				 * @constructor
				 *
				 * @ref opencv2/xfeatures2d/cuda.hpp -> cv::cuda::SURF_CUDA::create
				 */
				explicit SURFFeatureDetection(double hessianThreshold = 500, int nOctaves = 4,
											  int nOctaveLayers = 2, bool extended = false,
											  float keypointsRatio = 0.01f,
											  bool upright = false);

				/**
				 * @destructor
				 */
				~SURFFeatureDetection() override = default;

			};
		}
	}
}

#endif //CAMERASTABILIZATION_SURFFEATUREDETECTION_HPP
