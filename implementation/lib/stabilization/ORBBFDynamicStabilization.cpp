//
// Created by brucknem on 11.04.21.
//

#include "ORBBFDynamicStabilization.hpp"
#include "BruteForceFeatureMatching.hpp"
#include "ORBFeatureDetection.hpp"

namespace providentia {
	namespace stabilization {
		ORBBFDynamicStabilization::ORBBFDynamicStabilization(
			providentia::stabilization::detection::ORBFeatureDetection::Options options) {
			auto detector = providentia::stabilization::detection::ORBFeatureDetection(options);
			frameFeatureDetector = std::make_shared<providentia::stabilization::detection::ORBFeatureDetection>(
				detector);
			referenceFeatureDetector = std::make_shared<providentia::stabilization::detection::ORBFeatureDetection>(
				detector);
			matcher = std::make_shared<providentia::stabilization::features::BruteForceFeatureMatching>(
				cv::NORM_HAMMING);
			setName(typeid(*this).name());
		}
	}
}