//
// Created by brucknem on 11.04.21.
//

#ifndef CAMERASTABILIZATION_BRUTEFORCEFEATUREMATCHER_HPP
#define CAMERASTABILIZATION_BRUTEFORCEFEATUREMATCHER_HPP

#include "opencv2/opencv.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "FeatureDetectionBase.hpp"
#include "TimeMeasurable.hpp"
#include "FeatureMatchingBase.hpp"

namespace providentia {
	namespace stabilization {
		namespace features {
/**
 * Brute force feature matching on GPU.
 */
			class BruteForceFeatureMatcher : public providentia::stabilization::features::FeatureMatchingBase {
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
				 * @param goodMatchRatioThreshold The ratio threshold of good matches for the Lowe's ratio toCameraSpace.
				 */
				explicit BruteForceFeatureMatcher(cv::NormTypes norm, float goodMatchRatioThreshold = 0.75f);

			};
		}
	}
}

#endif //CAMERASTABILIZATION_BRUTEFORCEFEATUREMATCHER_HPP
