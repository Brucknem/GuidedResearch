//
// Created by brucknem on 11.04.21.
//

#include "FastFREAKBFDynamicStabilization.hpp"
#include "BruteForceFeatureMatching.hpp"
#include "FastFREAKFeatureDetection.hpp"

namespace providentia {
	namespace stabilization {
		FastFREAKBFDynamicStabilization::FastFREAKBFDynamicStabilization(int threshold,
																   bool nonmaxSuppression,
																   cv::FastFeatureDetector::DetectorType type,
																   int maxNPoints,
																   bool orientationNormalized,
																   bool scaleNormalized,
																   float patternScale,
																   int nOctaves,
																   const std::vector<int> &selectedPairs) {
			auto detector = providentia::stabilization::detection::FastFREAKFeatureDetection(threshold,
																							nonmaxSuppression, type,
																							maxNPoints,
																							orientationNormalized,
																							scaleNormalized,
																							patternScale, nOctaves,
																							selectedPairs);
			frameFeatureDetector = std::make_shared<providentia::stabilization::detection::FastFREAKFeatureDetection>(
				detector);
			referenceFeatureDetector = std::make_shared<providentia::stabilization::detection::FastFREAKFeatureDetection>(
				detector);
			matcher = std::make_shared<providentia::stabilization::features::BruteForceFeatureMatching>(
				cv::NORM_HAMMING);
			setName(typeid(*this).name());
		}
	}
}