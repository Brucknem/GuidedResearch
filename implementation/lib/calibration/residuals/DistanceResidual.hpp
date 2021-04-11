//
// Created by brucknem on 04.02.21.
//

#ifndef CAMERASTABILIZATION_DISTANCERESIDUAL_HPP
#define CAMERASTABILIZATION_DISTANCERESIDUAL_HPP

#include "Eigen/Dense"
#include "ceres/ceres.h"

#include "../camera/RenderingPipeline.hpp"
#include "../objects/WorldObject.hpp"

namespace providentia {
	namespace calibration {
		namespace residuals {

			class DistanceResidual {
			protected:
				double expectedValue;

			public:
				explicit DistanceResidual(double expectedValue);

				virtual ~DistanceResidual() = default;

				template<typename T>
				bool operator()(const T *value, T *residual) const;

				static ceres::CostFunction *create(double expectedValue);

			};
		}
	}
}

#endif //CAMERASTABILIZATION_DISTANCERESIDUAL_HPP
