//
// Created by brucknem on 11.04.21.
//

#ifndef CAMERASTABILIZATION_SURFBFDYNAMICSTABILIZER_HPP
#define CAMERASTABILIZATION_SURFBFDYNAMICSTABILIZER_HPP
#include "DynamicStabilizationBase.hpp"

namespace providentia {
	namespace stabilization {
/**
 * SURF feature detection and Brute Force feature matching stabilization algorithm.
 */
		class SURFBFDynamicStabilizer : public providentia::stabilization::DynamicStabilizationBase {
		public:
			/**
			 * @constructor
			 */
			explicit SURFBFDynamicStabilizer(double hessianThreshold = 1000, int nOctaves = 4,
											 int nOctaveLayers = 2, bool extended = false,
											 float keypointsRatio = 0.01f,
											 bool upright = false);

			/**
			 * @destructor
			 */
			~SURFBFDynamicStabilizer() override = default;
		};


	}}
#endif //CAMERASTABILIZATION_SURFBFDYNAMICSTABILIZER_HPP
