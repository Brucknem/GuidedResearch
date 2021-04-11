//
// Created by brucknem on 04.02.21.
//

#ifndef CAMERASTABILIZATION_RESIDUALS_HPP
#define CAMERASTABILIZATION_RESIDUALS_HPP

#include "Eigen/Dense"
#include "ceres/ceres.h"

#include "RenderingPipeline.hpp"
#include "WorldObjects.hpp"

namespace providentia {
	namespace calibration {

		class DistanceResidual {
		protected:
			double expectedValue;

		public:
			explicit DistanceResidual(double expectedValue);

			template<typename T>
			bool operator()(const T *value, T *residual) const;

			static ceres::CostFunction *Create(double expectedValue);

		};

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
			static ceres::CostFunction *Create(double lowerBound, double upperBound);

			static ceres::CostFunction *Create(double upperBound);
		};

		/**
		 * A residual term used in the optimization process to find the camera pose.
		 */
		class CorrespondenceResidual {
		protected:
			/**
			 * The expected [u, v] pixel corresponding to the world coordinate.
			 */
			Eigen::Vector2d expectedPixel;

			/**
			 * The intrinsics matrix of the pinhole camera model.
			 */
			Eigen::Matrix<double, 3, 4> intrinsics;

			/**
			 * The parametricPoint that contains the correspondence.
			 */
			providentia::calibration::ParametricPoint parametricPoint;

		public:
			/**
			 * @constructor
			 *
			 * @param expectedPixel The expected [u, v] pixel location.
			 * @param point The [x, y, z] point that corresponds to the pixel.
			 * @param intrinsics The intrinsics of the pinhole camera model.
			 * @param imageSize The [width, height] of the image.
			 */
			CorrespondenceResidual(Eigen::Vector2d expectedPixel,
								   const providentia::calibration::ParametricPoint &point,
								   Eigen::Matrix<double, 3, 4> intrinsics);

			/**
			 * @destructor
			 */
			virtual ~CorrespondenceResidual() = default;

			/**
			 * Calculates the residual error after transforming the world position to a pixel.
			 *
			 * @tparam T Template parameter expected from the ceres-solver.
			 * @param translation The [x, y, z] translation of the camera in world space for which we optimize.
			 * @param rotation The [x, y, z] euler angle rotation of the camera around the world axis for which we optimize.
			 * @param lambda The [l] distance of the point in the direction of one side of the parametricPoint from the origin.
			 * @param mu The [m] distance of the point in the direction of another side of the parametricPoint from the origin.
			 * @param residual The [u, v] pixel error between the expected and calculated pixel.
			 * @return true
			 */
			template<typename T>
			bool operator()(
				const T *tx,
				const T *ty,
				const T *tz,
				const T *rx,
				const T *ry,
				const T *rz,
				const T *lambda,
				const T *mu,
				const T *weight,
				T *residual)
			const;

			/**
			 * Factory method to hide the residual creation.
			 */
			static ceres::CostFunction *Create(const Eigen::Vector2d &expectedPixel,
											   const providentia::calibration::ParametricPoint &point,
											   const Eigen::Matrix<double, 3, 4> &intrinsics);
		};

	}
}

#endif //CAMERASTABILIZATION_RESIDUALS_HPP
