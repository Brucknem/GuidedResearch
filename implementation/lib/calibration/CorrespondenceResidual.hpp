//
// Created by brucknem on 11.04.21.
//

#ifndef CAMERASTABILIZATION_CORRESPONDENCERESIDUAL_HPP
#define CAMERASTABILIZATION_CORRESPONDENCERESIDUAL_HPP

#include <eigen3/Eigen/src/Core/Matrix.h>
#include "WorldObject.hpp"

namespace providentia {
	namespace calibration {
		namespace residuals {

			/**
			 * A residual term used in the optimization process to find the camera pose.
			 */
			class CorrespondenceResidual {
			protected:
				/**
				 * The expected [u, v] pixel corresponding to the world coordinate.
				 */
				Eigen::Matrix<double, 2, 1> expectedPixel;

				/**
				 * The intrinsics matrix of the pinhole camera model.
				 */
				Eigen::Matrix<double, 3, 4> intrinsics;

				/**
				 * The parametricPoint that contains the correspondence.
				 */
				ParametricPoint parametricPoint;

			public:
				/**
				 * @constructor
				 *
				 * @param expectedPixel The expected [u, v] pixel location.
				 * @param point The [x, y, z] point that corresponds to the pixel.
				 * @param intrinsics The intrinsics of the pinhole camera model.
				 * @param imageSize The [width, height] of the image.
				 */
				CorrespondenceResidual(Eigen::Matrix<double, 2, 1> expectedPixel,
									   const ParametricPoint &point,
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
				static ceres::CostFunction *create(const Eigen::Matrix<double, 2, 1> &expectedPixel,
												   const ParametricPoint &point,
												   const Eigen::Matrix<double, 3, 4> &intrinsics);
			};
		}
	}
}

#include "Eigen/Dense"
#include "ceres/ceres.h"
#include "RenderingPipeline.hpp"
#include "WorldObject.hpp"
#include "ParametricPoint.hpp"

#endif //CAMERASTABILIZATION_CORRESPONDENCERESIDUAL_HPP
