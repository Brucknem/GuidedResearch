//
// Created by brucknem on 11.04.21.
//

#ifndef CAMERASTABILIZATION_SURFBFDYNAMICSTABILIZATION_HPP
#define CAMERASTABILIZATION_SURFBFDYNAMICSTABILIZATION_HPP

#include "DynamicStabilizationBase.hpp"

namespace providentia {
	namespace stabilization {
/**
 * SURF feature detection and Brute Force feature matching stabilization algorithm.
 */
		class SURFBFDynamicStabilization : public providentia::stabilization::DynamicStabilizationBase {
		public:
			/**
			 * @constructor
			 */
			explicit SURFBFDynamicStabilization(double hessianThreshold = 1000, int nOctaves = 4,
												int nOctaveLayers = 2, bool extended = false,
												float keypointsRatio = 0.01f,
												bool upright = false);

			/**
			 * @destructor
			 */
			~SURFBFDynamicStabilization() override = default;
		};

	}
}
#endif //CAMERASTABILIZATION_SURFBFDYNAMICSTABILIZATION_HPP
