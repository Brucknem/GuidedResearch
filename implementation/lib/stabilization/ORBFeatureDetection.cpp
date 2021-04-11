//
// Created by brucknem on 11.04.21.
//

#include "ORBFeatureDetection.hpp"

namespace providentia {
	namespace stabilization {
		namespace detection {

			ORBFeatureDetection::ORBFeatureDetection(int nfeatures, float scaleFactor, int nlevels,
												   int edgeThreshold, int firstLevel, int wtaK,
												   int scoreType, int patchSize, int fastThreshold,
												   bool blurForDescriptor) {
				detector = cv::cuda::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, wtaK,
												 scoreType, patchSize, fastThreshold, blurForDescriptor);
				providentia::utils::TimeMeasurable::setName(typeid(*this).name());
			}

			void ORBFeatureDetection::specificDetect() {
				detector->detectAndComputeAsync(processedFrame, getCurrentMask(processedFrame.size()), keypointsGPU,
												descriptorsGPU, false);
				detector->convert(keypointsGPU, keypointsCPU);
				descriptorsGPU.download(descriptorsCPU);
			}
		}
	}
}