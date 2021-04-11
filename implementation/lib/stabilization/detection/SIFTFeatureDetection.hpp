//
// Created by brucknem on 11.04.21.
//

#ifndef CAMERASTABILIZATION_SIFTFEATUREDETECTION_HPP
#define CAMERASTABILIZATION_SIFTFEATUREDETECTION_HPP

#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include <opencv2/cudafeatures2d.hpp>

#include "../TimeMeasurable.hpp"
#include "FeatureDetectionBase.hpp"

namespace providentia {
	namespace stabilization {
		namespace detection {

/**
 * Wrapper for the CUDA SURF feature detector.
 */
			class SIFTFeatureDetection : public providentia::stabilization::detection::FeatureDetectionBase {

			private:
				/**
				 * The CPU SIFT detector used to detect keypoints and descriptors.
				 */
				cv::Ptr<cv::SIFT> detector;

			protected:
				/**
				 * @copydoc
				 */
				void specificDetect() override;

			public:
				struct Options {
					int nfeatures = 0;
					int nOctaveLayers = 3;
					double contrastThreshold = 0.04;
					double edgeThreshold = 10;
					double sigma = 1.6;

					Options() {
						// Necessary for default initialization
					}
				};

				/**
				 * @constructor
				 *
				 * @ref opencv2/xfeatures2d/cuda.hpp -> cv::cuda::SURF_CUDA::create
				 */
				explicit SIFTFeatureDetection(Options options = {});

				/**
				 * @destructor
				 */
				~SIFTFeatureDetection() override = default;
			};

		}
	}
}
#endif //CAMERASTABILIZATION_SIFTFEATUREDETECTION_HPP
