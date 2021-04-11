//
// Created by brucknem on 11.04.21.
//

#ifndef CAMERASTABILIZATION_ORBFEATUREDETECTION_HPP
#define CAMERASTABILIZATION_ORBFEATUREDETECTION_HPP

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
			class ORBFeatureDetection : public providentia::stabilization::detection::FeatureDetectionBase {
			private:
				/**
				 * The CUDA ORB detector used to detect keypoints and descriptors.
				 */
				cv::Ptr<cv::cuda::ORB> detector;

			protected:
				/**
				 * @copydoc
				 */
				void specificDetect() override;

			public:
				struct Options {
					int nfeatures = 1e4;
					float scaleFactor = 1.2f;
					int nlevels = 8;
					int edgeThreshold = 31;
					int firstLevel = 0;
					int wtaK = 2;
					int scoreType = cv::ORB::FAST_SCORE;
					int patchSize = 31;
					int fastThreshold = 20;
					bool blurForDescriptor = false;

					Options() {}
				};

				/**
				 * @constructor
				 *
				 * @ref opencv2/cudafeatures2d.hpp -> cv::cuda::ORB::create
				 */
				explicit ORBFeatureDetection(Options options = Options());

				/**
				 * @destructor
				 */
				~ORBFeatureDetection() override = default;
			};

		}
	}
}
#endif //CAMERASTABILIZATION_ORBFEATUREDETECTION_HPP
