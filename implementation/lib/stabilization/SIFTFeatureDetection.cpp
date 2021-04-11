//
// Created by brucknem on 11.04.21.
//

#include "SIFTFeatureDetection.hpp"

namespace providentia {
	namespace stabilization {
		namespace detection {

			SIFTFeatureDetection::SIFTFeatureDetection(int nfeatures, int nOctaveLayers,
													 double contrastThreshold, double edgeThreshold,
													 double sigma) {
				detector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
				setName(typeid(*this).name());
			}

			void SIFTFeatureDetection::specificDetect() {
				detector->detectAndCompute(frameCPU, cv::Mat(getCurrentMask()), keypointsCPU, descriptorsCPU);
				descriptorsGPU.upload(descriptorsCPU);
			}
		}
	}
}