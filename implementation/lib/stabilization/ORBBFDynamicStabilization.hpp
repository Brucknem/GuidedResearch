//
// Created by brucknem on 11.04.21.
//

#ifndef CAMERASTABILIZATION_ORBBFDYNAMICSTABILIZATION_HPP
#define CAMERASTABILIZATION_ORBBFDYNAMICSTABILIZATION_HPP

#include "DynamicStabilizationBase.hpp"

namespace providentia {
	namespace stabilization {
/**
 * ORB feature detection and Brute Force feature matching stabilization algorithm.
 */
		class ORBBFDynamicStabilization : public providentia::stabilization::DynamicStabilizationBase {
		public:
			/**
			 * @constructor
			 */
			explicit ORBBFDynamicStabilization(int nfeatures = 1e4,
											   float scaleFactor = 1.2f,
											   int nlevels = 8,
											   int edgeThreshold = 31,
											   int firstLevel = 0,
											   int wtaK = 2,
											   int scoreType = cv::ORB::FAST_SCORE,
											   int patchSize = 31,
											   int fastThreshold = 20,
											   bool blurForDescriptor = false);

			/**
			 * @destructor
			 */
			~ORBBFDynamicStabilization() override = default;
		};
	}
}

#endif //CAMERASTABILIZATION_ORBBFDYNAMICSTABILIZATION_HPP
