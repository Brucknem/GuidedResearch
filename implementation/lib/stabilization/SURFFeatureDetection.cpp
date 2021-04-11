//
// Created by brucknem on 11.04.21.
//

#include "SURFFeatureDetection.hpp"

namespace providentia {
	namespace stabilization {
		namespace detection {

			SURFFeatureDetection::SURFFeatureDetection(double hessianThreshold, int nOctaves,
													 int nOctaveLayers, bool extended,
													 float keypointsRatio, bool upright) {
				detector = cv::cuda::SURF_CUDA::create(hessianThreshold, nOctaves, nOctaveLayers, extended,
													   keypointsRatio, upright);
				providentia::utils::TimeMeasurable::setName(typeid(*this).name());
			}

			void SURFFeatureDetection::specificDetect() {
				detector->detectWithDescriptors(processedFrame, getCurrentMask(processedFrame.size()), keypointsGPU,
												descriptorsGPU, false);
				detector->downloadKeypoints(keypointsGPU, keypointsCPU);
				descriptorsGPU.download(descriptorsCPU);
			}
		}
	}
}