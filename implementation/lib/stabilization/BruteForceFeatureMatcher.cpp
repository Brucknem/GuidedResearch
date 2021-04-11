//
// Created by brucknem on 11.04.21.
//

#include "BruteForceFeatureMatcher.hpp"

namespace providentia {
	namespace stabilization {
		namespace features {
			BruteForceFeatureMatcher::BruteForceFeatureMatcher(cv::NormTypes norm,
															   float goodMatchRatioThreshold)
				: FeatureMatchingBase(goodMatchRatioThreshold) {
				matcher = cv::cuda::DescriptorMatcher::createBFMatcher(norm);
				setName(typeid(*this).name());
			}

			void BruteForceFeatureMatcher::specificMatch() {
				if (frameDetector->getDescriptorsGPU().empty() || frameDetector->getDescriptorsGPU().empty()) {
					throw std::invalid_argument("Possibly match with wrong descriptor format called.");
				}
				matcher->knnMatchAsync(frameDetector->getDescriptorsGPU(), referenceFrameDetector->getDescriptorsGPU(),
									   knnMatchesGPU, 2, cv::noArray(), stream);
				stream.waitForCompletion();
				matcher->knnMatchConvert(knnMatchesGPU, knnMatchesCPU);
			}
		}
	}
}