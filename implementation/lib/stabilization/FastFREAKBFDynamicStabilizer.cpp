//
// Created by brucknem on 11.04.21.
//

#include "FastFREAKBFDynamicStabilizer.hpp"
#include "BruteForceFeatureMatcher.hpp"

namespace providentia {
	namespace stabilization {
		FastFREAKBFDynamicStabilizer::FastFREAKBFDynamicStabilizer(int threshold,
																   bool nonmaxSuppression,
																   cv::FastFeatureDetector::DetectorType type,
																   int maxNPoints,
																   bool orientationNormalized,
																   bool scaleNormalized,
																   float patternScale,
																   int nOctaves,
																   const std::vector<int> &selectedPairs) {
			auto detector = providentia::stabilization::features::FastFREAKFeatureDetector(threshold,
																						   nonmaxSuppression, type,
																						   maxNPoints,
																						   orientationNormalized,
																						   scaleNormalized,
																						   patternScale, nOctaves,
																						   selectedPairs);
			frameFeatureDetector = std::make_shared<providentia::stabilization::features::FastFREAKFeatureDetector>(
				detector);
			referenceFeatureDetector = std::make_shared<providentia::stabilization::features::FastFREAKFeatureDetector>(
				detector);
			matcher = std::make_shared<providentia::stabilization::features::BruteForceFeatureMatcher>(
				cv::NORM_HAMMING);
			setName(typeid(*this).name());
		}
	}
}