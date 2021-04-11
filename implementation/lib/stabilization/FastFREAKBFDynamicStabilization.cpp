//
// Created by brucknem on 11.04.21.
//

#include "FastFREAKBFDynamicStabilization.hpp"
#include "BruteForceFeatureMatching.hpp"
#include "FastFREAKFeatureDetection.hpp"

namespace providentia {
	namespace stabilization {
		FastFREAKBFDynamicStabilization::FastFREAKBFDynamicStabilization
			(providentia::stabilization::detection::FastFREAKFeatureDetection::Options options) {
			auto detector = providentia::stabilization::detection::FastFREAKFeatureDetection(options);
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