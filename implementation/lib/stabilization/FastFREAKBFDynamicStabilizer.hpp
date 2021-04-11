//
// Created by brucknem on 11.04.21.
//

#ifndef CAMERASTABILIZATION_FASTFREAKBFDYNAMICSTABILIZER_HPP
#define CAMERASTABILIZATION_FASTFREAKBFDYNAMICSTABILIZER_HPP

#include "DynamicStabilizationBase.hpp"

namespace providentia {
	namespace stabilization {
/**
 * Dynamic stabilization with Fast feature detectors, FREAK feature descriptors and Brute Force matching.
 */
		class FastFREAKBFDynamicStabilizer : public providentia::stabilization::DynamicStabilizationBase {
		public:
			/**
			 * @constructor
			 */
			explicit FastFREAKBFDynamicStabilizer(int threshold = 50,
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
			~FastFREAKBFDynamicStabilizer() override = default;
		};
	}
}

#endif //CAMERASTABILIZATION_FASTFREAKBFDYNAMICSTABILIZER_HPP
