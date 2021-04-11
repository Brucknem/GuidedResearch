//
// Created by brucknem on 11.04.21.
//

#ifndef CAMERASTABILIZATION_FLANNFEATUREMATCHER_HPP
#define CAMERASTABILIZATION_FLANNFEATUREMATCHER_HPP

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
 * CPU implementation of the FLANN feature matcher.
 */
			class FlannFeatureMatcher : public providentia::stabilization::features::FeatureMatchingBase {
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
				 * @param goodMatchRatioThreshold The ratio threshold of good matches for the Lowe's ratio toCameraSpace.
				 */
				explicit FlannFeatureMatcher(bool binaryDescriptors = false,
											 float goodMatchRatioThreshold = 0.75f);

				/**
				 * @constructor
				 *
				 * @params The FLANN parameters.
				 * @param goodMatchRatioThreshold The ratio threshold of good matches for the Lowe's ratio toCameraSpace.
				 */
				explicit FlannFeatureMatcher(cv::flann::IndexParams *params, float goodMatchRatioThreshold = 0.75f);

			};
		}
	}
}

#endif //CAMERASTABILIZATION_FLANNFEATUREMATCHER_HPP
