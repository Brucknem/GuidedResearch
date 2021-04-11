//
// Created by brucknem on 04.02.21.
//

#ifndef CAMERASTABILIZATION_DISTANCEFROMINTERVALRESIDUAL_HPP
#define CAMERASTABILIZATION_DISTANCEFROMINTERVALRESIDUAL_HPP

#include "Eigen/Dense"
#include "ceres/ceres.h"

#include "RenderingPipeline.hpp"
#include "WorldObjects.hpp"

namespace providentia {
	namespace calibration {
		namespace residuals {

			class DistanceFromIntervalResidual {
			protected:
				double lowerBound;
				double upperBound;

			public:
				explicit DistanceFromIntervalResidual(double upperBound);

				DistanceFromIntervalResidual(double lowerBound, double upperBound);

				template<typename T>
				bool operator()(const T *value, T *residual) const;

				/**
				 * Factory method to hide the residual creation.
				 */
				static ceres::CostFunction *create(double lowerBound, double upperBound);

				static ceres::CostFunction *create(double upperBound);
			};

		}
	}
}

#endif //CAMERASTABILIZATION_DISTANCEFROMINTERVALRESIDUAL_HPP
