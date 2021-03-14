//
// Created by brucknem on 04.02.21.
//

#ifndef CAMERASTABILIZATION_RESIDUALS_HPP
#define CAMERASTABILIZATION_RESIDUALS_HPP

#include "Eigen/Dense"
#include "ceres/ceres.h"

#include "RenderingPipeline.hpp"
#include "CameraPoseEstimation.hpp"
#include "WorldObjects.hpp"

namespace providentia {
	namespace calibration {

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
			std::shared_ptr<providentia::calibration::ParametricPoint> parametricPoint;

			/**
			 * The weight factor of the residual.
			 */
			double weight;

			/**
			 * The maximal valid lambda value, i.e. the height of the object.
			 */
			double maxLambda;

			/**
			 * The maximal valid lambda value, i.e. the height of the object.
			 */
			double maxMu;

			/**
			 * Renders the given point using the current given camera translation and rotation.
			 *
			 * @tparam T Template parameter expected from the ceres-solver.
			 * @param _translation The [x, y, z] translation of the camera in world space for which we optimize.
			 * @param _rotation The [x, y, z] euler angle rotation of the camera around the world axis for which we optimize.
			 * @param _point The [x, y, z] world position vector.
			 * @param residual The [u, v] pixel error between the expected and calculated pixel.
			 */
			template<typename T>
			bool calculateResidual(const T *_translation, const T *_rotation, const T *_point, T *residual) const;

		public:
			/**
			 * @constructor
			 *
			 * @param _expectedPixel The expected [u, v] pixel location.
			 * @param _point The [x, y, z] point that corresponds to the pixel.
			 * @param _intrinsics The intrinsics of the pinhole camera model.
			 * @param _imageSize The [width, height] of the image.
			 */
			CorrespondenceResidual(Eigen::Vector2d _expectedPixel,
								   std::shared_ptr<providentia::calibration::ParametricPoint> _point,
								   Eigen::Matrix<double, 3, 4> _intrinsics,
								   double maxLambda,
								   double maxMu,
								   double _weight);

			/**
			 * @destructor
			 */
			virtual ~CorrespondenceResidual() = default;

			/**
			 * Calculates the residual error after transforming the world position to a pixel.
			 *
			 * @tparam T Template parameter expected from the ceres-solver.
			 * @param _translation The [x, y, z] translation of the camera in world space for which we optimize.
			 * @param _rotation The [x, y, z] euler angle rotation of the camera around the world axis for which we optimize.
			 * @param _lambda The [l] distance of the point in the direction of one side of the parametricPoint from the origin.
			 * @param _mu The [m] distance of the point in the direction of another side of the parametricPoint from the origin.
			 * @param residual The [u, v] pixel error between the expected and calculated pixel.
			 * @return true
			 */
			template<typename T>
			bool operator()(const T *_translation, const T *_rotation, const T *_lambda, const T *_mu, const T *
			angle, T *residual)
			const;

			/**
			 * Factory method to hide the residual creation.
			 */
			static ceres::CostFunction *Create(const Eigen::Vector2d &_expectedPixel,
											   const std::shared_ptr<providentia::calibration::ParametricPoint> &point,
											   const Eigen::Matrix<double, 3, 4> &intrinsics,
											   double maxLambda,
											   double maxMu,
											   double weight);
		};

	}
}

#endif //CAMERASTABILIZATION_RESIDUALS_HPP
