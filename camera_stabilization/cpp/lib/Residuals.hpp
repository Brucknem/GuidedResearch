//
// Created by brucknem on 04.02.21.
//

#ifndef CAMERASTABILIZATION_RESIDUALS_HPP
#define CAMERASTABILIZATION_RESIDUALS_HPP

#include "Eigen/Dense"
#include "ceres/ceres.h"

#include "RenderingPipeline.hpp"
#include "CameraPoseEstimation.hpp"

namespace providentia {
	namespace calibration {

		/**
		 * Represents the residual block when calculating the reprojection error of a known world position to a pixel.
		 */
		struct PointCorrespondenceResidual {
		private:
			/**
			 * The expected pixel corresponding to the world coordinate.
			 */
			Eigen::Vector2d expectedPixel;

			/**
			 * The given world position that results in the expected pixel.
			 */
			Eigen::Vector3d worldPosition;

			/**
			 * The [near, far] plane distances of the view frustum.
			 */
			Eigen::Vector2d frustumParameters;

			/**
			 * The [sensorWidth, aspectRatio, focalLength] of the pinhole camera model.
			 */
			Eigen::Vector3d intrinsics;

			/**
			 * The [width, height] of the image.
			 */
			Eigen::Vector2d imageSize;

		public:
			/**
			 * @constructor
			 *
			 * @param _expectedPixel The expected pixel location.
			 * @param _worldPosition The given location in world space.
			 * @param _frustumParameters The [near, far] plane distances of the view frustum.
			 * @param _intrinsics The [sensorWidth, aspectRatio, focalLength] of the pinhole camera model.
			 * @param _imageSize The [width, height] of the image.
			 */
			PointCorrespondenceResidual(Eigen::Vector2d _expectedPixel, Eigen::Vector3d _worldPosition,
										Eigen::Vector2d _frustumParameters, Eigen::Vector3d _intrinsics,
										Eigen::Vector2d _imageSize);

			/**
			 * Calculates the residual error after transforming the world position to a pixel.
			 *
			 * @tparam T Template parameter expected from the ceres-solver.
			 * @param _translation The [x, y, z] translation of the camera in world space for which we optimize.
			 * @param _rotation The [x, y, z] euler angle rotation of the camera around the world axis for which we optimize.
			 * @param residual The [u, v] pixel error between the expected and calculated pixel.
			 * @return true
			 */
			template<typename T>
			bool operator()(const T *_translation, const T *_rotation, T *residual) const;

			/**
			 * Factory method to hide the residual creation.
			 */
			static ceres::CostFunction *Create(Eigen::Vector2d _expectedPixel, Eigen::Vector3d _worldPosition,
											   Eigen::Vector2d _frustumParameters, Eigen::Vector3d _intrinsics,
											   Eigen::Vector2d _imageSize);

		};

	}
}

#endif //CAMERASTABILIZATION_RESIDUALS_HPP
