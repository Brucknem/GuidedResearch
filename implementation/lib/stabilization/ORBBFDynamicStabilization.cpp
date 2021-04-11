//
// Created by brucknem on 11.04.21.
//

#include "ORBBFDynamicStabilization.hpp"
#include "BruteForceFeatureMatching.hpp"
#include "ORBFeatureDetection.hpp"

namespace providentia {
	namespace stabilization {
		ORBBFDynamicStabilization::ORBBFDynamicStabilization(int nfeatures, float scaleFactor,
													   int nlevels, int edgeThreshold,
													   int firstLevel, int wtaK,
													   int scoreType,
													   int patchSize, int fastThreshold,
													   bool blurForDescriptor) {
			auto detector = providentia::stabilization::detection::ORBFeatureDetection(nfeatures, scaleFactor,
																					  nlevels, edgeThreshold,
																					  firstLevel, wtaK, scoreType,
																					  patchSize, fastThreshold,
																					  blurForDescriptor);
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