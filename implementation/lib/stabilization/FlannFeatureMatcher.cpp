//
// Created by brucknem on 11.04.21.
//

#include "FlannFeatureMatcher.hpp"

namespace providentia {
	namespace stabilization {
		namespace features {
			FlannFeatureMatcher::FlannFeatureMatcher(cv::flann::IndexParams *params,
													 float goodMatchRatioThreshold)
				: FeatureMatchingBase(
				goodMatchRatioThreshold) {
				matcher = std::make_shared<cv::FlannBasedMatcher>(params);
				setName(typeid(*this).name());
			}

			void FlannFeatureMatcher::specificMatch() {
				if (frameDetector->getDescriptorsCPU().empty() || referenceFrameDetector->getDescriptorsCPU().empty()) {
					throw std::invalid_argument("Possibly match with wrong descriptor format called.");
				}
				matcher->knnMatch(frameDetector->getDescriptorsCPU(), referenceFrameDetector->getDescriptorsCPU(),
								  knnMatchesCPU,
								  2);
			}

			FlannFeatureMatcher::FlannFeatureMatcher(bool binaryDescriptors,
													 float goodMatchRatioThreshold) : FeatureMatchingBase(
				goodMatchRatioThreshold) {
				if (binaryDescriptors) {
					matcher = std::make_shared<cv::FlannBasedMatcher>(new cv::flann::LshIndexParams(12, 20, 2));
				} else {
					matcher = std::make_shared<cv::FlannBasedMatcher>();
				}
			}
		}
	}
}