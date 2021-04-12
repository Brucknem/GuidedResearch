//
// Created by brucknem on 04.02.21.
//

#ifndef CAMERASTABILIZATION_DISTANCEFROMINTERVALRESIDUAL_HPP
#define CAMERASTABILIZATION_DISTANCEFROMINTERVALRESIDUAL_HPP

#include "Eigen/Dense"
#include "ceres/ceres.h"

#include "../camera/RenderingPipeline.hpp"
#include "../objects/WorldObject.hpp"

namespace providentia {
	namespace calibration {
		namespace residuals {

			/**
			 * Residual for the distance of a value to a given fixed interval.
			 *
			 * 							|	x - u 	if x > u
			 * Minimizes: f(x, l, u) = 	|	x - l	if x < l
			 * 							|	0 		else
			 */
			class DistanceFromIntervalResidual {
			protected:

				/**
				 * The lower bound if the interval.
				 */
				double lowerBound;

				/**
				 * The upper bound if the interval.
				 */
				double upperBound;

			public:

				/**
				 * @constructor
				 *
				 * @param upperBound The upper bound of the interval, lower bound is 0.
				 */
				explicit DistanceFromIntervalResidual(double upperBound);

				/**
				 * @constructor
				 *
				 * @param lowerBound The lower bound of the interval.
				 * @param upperBound The upper bound of the interval.
				 */
				DistanceFromIntervalResidual(double lowerBound, double upperBound);

				/**
				 * @destructor
				 */
				virtual ~DistanceFromIntervalResidual() = default;

				/**
				 * Residual calculation function.
				 *
				 * @tparam T double or ceres::Jet<double, 1>
				 *
				 * @param value The current estimated value.
				 * @param residual The residual, i.e. f(x, l, u)
				 *
				 * @return true
				 */
				template<typename T>
				bool operator()(const T *value, T *residual) const;

				/**
				 * Factory method to ease residual creation.
				 *
				 * @param upperBound The upper bound of the interval, lower bound is 0.
				 *
				 * @return The cost function based on the residual.
				 */
				static ceres::CostFunction *create(double upperBound);

				/**
				 * Factory method to ease residual creation.
				 *
				 * @param lowerBound The lower bound of the interval.
				 * @param upperBound The upper bound of the interval.
				 *
				 * @return The cost function based on the residual.
				 */
				static ceres::CostFunction *create(double lowerBound, double upperBound);
			};

		}
	}
}

#endif //CAMERASTABILIZATION_DISTANCEFROMINTERVALRESIDUAL_HPP
