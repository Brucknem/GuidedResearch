//
// Created by brucknem on 11.04.21.
//

#include "FastFREAKFeatureDetection.hpp"

namespace providentia {
	namespace stabilization {
		namespace detection {
			FastFREAKFeatureDetection::FastFREAKFeatureDetection(int threshold, bool nonmaxSuppression,
															   cv::FastFeatureDetector::DetectorType type,
															   int maxNPoints,
															   bool orientationNormalized,
															   bool scaleNormalized,
															   float patternScale,
															   int nOctaves,
															   const std::vector<int> &selectedPairs) {
				detector = cv::cuda::FastFeatureDetector::create(threshold, nonmaxSuppression, type, maxNPoints);
				descriptor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale,
															nOctaves,
															selectedPairs);
				providentia::utils::TimeMeasurable::setName(typeid(*this).name());
			}

			void FastFREAKFeatureDetection::specificDetect() {
				detector->detect(processedFrame, keypointsCPU, getCurrentMask(processedFrame.size()));
				descriptor->compute(frameCPU, keypointsCPU, descriptorsCPU);
				descriptorsGPU.upload(descriptorsCPU);
			}
		}
	}
}