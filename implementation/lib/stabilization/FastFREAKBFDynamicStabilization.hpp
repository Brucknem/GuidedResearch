//
// Created by brucknem on 11.04.21.
//

#ifndef CAMERASTABILIZATION_FASTFREAKBFDYNAMICSTABILIZATION_HPP
#define CAMERASTABILIZATION_FASTFREAKBFDYNAMICSTABILIZATION_HPP

#include "DynamicStabilizationBase.hpp"

namespace providentia {
	namespace stabilization {
/**
 * Dynamic stabilization with Fast feature detectors, FREAK feature descriptors and Brute Force matching.
 */
		class FastFREAKBFDynamicStabilization : public providentia::stabilization::DynamicStabilizationBase {
		public:
			/**
			 * @constructor
			 */
			explicit FastFREAKBFDynamicStabilization(int threshold = 50,
													 bool nonmaxSuppression = true,
													 cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16,
													 int maxNPoints = 5000,
													 bool orientationNormalized = true,
													 bool scaleNormalized = true,
													 float patternScale = 22.0f,
													 int nOctaves = 4,
													 const std::vector<int> &selectedPairs = std::vector<int>());

			/**
			 * @destructor
			 */
			~FastFREAKBFDynamicStabilization() override = default;
		};
	}
}

#endif //CAMERASTABILIZATION_FASTFREAKBFDYNAMICSTABILIZATION_HPP
