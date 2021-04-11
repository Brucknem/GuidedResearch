//
// Created by brucknem on 11.04.21.
//

#include "StarBRIEFFeatureDetection.hpp"

namespace providentia {
	namespace stabilization {
		namespace detection {

			StarBRIEFFeatureDetection::StarBRIEFFeatureDetection(int maxSize, int responseThreshold,
															   int lineThresholdProjected,
															   int lineThresholdBinarized,
															   int suppressNonmaxSize, int bytes,
															   bool useOrientation) {
				detector = cv::xfeatures2d::StarDetector::create(maxSize, responseThreshold, lineThresholdProjected,
																 lineThresholdBinarized, suppressNonmaxSize);
				descriptor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, useOrientation);
				providentia::utils::TimeMeasurable::setName(typeid(*this).name());
			}

			void StarBRIEFFeatureDetection::specificDetect() {
				detector->detect(FeatureDetectionBase::frameCPU, FeatureDetectionBase::keypointsCPU, cv::Mat(
					FeatureDetectionBase::getCurrentMask(FeatureDetectionBase::processedFrame.size())));
				descriptor->compute(FeatureDetectionBase::frameCPU, FeatureDetectionBase::keypointsCPU,
									FeatureDetectionBase::descriptorsCPU);
				FeatureDetectionBase::descriptorsGPU.upload(FeatureDetectionBase::descriptorsCPU);
			}
		}
	}
}