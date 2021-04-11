//
// Created by brucknem on 11.04.21.
//

#include "SURFBFDynamicStabilizer.hpp"

namespace providentia {
	namespace stabilization {
		SURFBFDynamicStabilizer::SURFBFDynamicStabilizer(double hessianThreshold,
														 int nOctaves,
														 int nOctaveLayers,
														 bool extended,
														 float keypointsRatio,
														 bool upright) {
			auto detector = providentia::features::SURFFeatureDetector(hessianThreshold, nOctaves,
																	   nOctaveLayers, extended,
																	   keypointsRatio, upright);
			providentia::stabilization::DynamicStabilizationBase::frameFeatureDetector = std::make_shared<providentia::features::SURFFeatureDetector>(
				detector);
			providentia::stabilization::DynamicStabilizationBase::referenceFeatureDetector = std::make_shared<providentia::features::SURFFeatureDetector>(
				detector);
			providentia::stabilization::DynamicStabilizationBase::matcher = std::make_shared<providentia::features::BruteForceFeatureMatcher>(
				cv::NORM_L2);
			providentia::utils::TimeMeasurable::setName(typeid(*this).name());
		}
	}
}