//
// Created by brucknem on 11.04.21.
//

#include "SURFBFDynamicStabilization.hpp"
#include "BruteForceFeatureMatching.hpp"
#include "SURFFeatureDetection.hpp"

namespace providentia {
	namespace stabilization {
		SURFBFDynamicStabilization::SURFBFDynamicStabilization(double hessianThreshold,
															   int nOctaves,
															   int nOctaveLayers,
															   bool extended,
															   float keypointsRatio,
															   bool upright) {
			auto detector = providentia::stabilization::detection::SURFFeatureDetection(hessianThreshold, nOctaves,
																					   nOctaveLayers, extended,
																					   keypointsRatio, upright);
			providentia::stabilization::DynamicStabilizationBase::frameFeatureDetector = std::make_shared<providentia::stabilization::detection::SURFFeatureDetection>(
				detector);
			providentia::stabilization::DynamicStabilizationBase::referenceFeatureDetector = std::make_shared<providentia::stabilization::detection::SURFFeatureDetection>(
				detector);
			providentia::stabilization::DynamicStabilizationBase::matcher = std::make_shared<providentia::stabilization::features::BruteForceFeatureMatching>(
				cv::NORM_L2);
			providentia::utils::TimeMeasurable::setName(typeid(*this).name());
		}
	}
}