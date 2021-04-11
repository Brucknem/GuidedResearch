//
// Created by brucknem on 11.04.21.
//

#include "ORBBFDynamicStabilizer.hpp"
#include "BruteForceFeatureMatcher.hpp"

namespace providentia {
	namespace stabilization {
		ORBBFDynamicStabilizer::ORBBFDynamicStabilizer(int nfeatures, float scaleFactor,
													   int nlevels, int edgeThreshold,
													   int firstLevel, int wtaK,
													   int scoreType,
													   int patchSize, int fastThreshold,
													   bool blurForDescriptor) {
			auto detector = providentia::stabilization::features::ORBFeatureDetector(nfeatures, scaleFactor,
																					 nlevels, edgeThreshold,
																					 firstLevel, wtaK, scoreType,
																					 patchSize, fastThreshold,
																					 blurForDescriptor);
			frameFeatureDetector = std::make_shared<providentia::stabilization::features::ORBFeatureDetector>(detector);
			referenceFeatureDetector = std::make_shared<providentia::stabilization::features::ORBFeatureDetector>(
				detector);
			matcher = std::make_shared<providentia::stabilization::features::BruteForceFeatureMatcher>(
				cv::NORM_HAMMING);
			setName(typeid(*this).name());
		}
	}
}