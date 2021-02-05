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

		class CorrespondenceResidualBase {
		protected:
			/**
			 * The expected [u, v] pixel corresponding to the world coordinate.
			 */
			Eigen::Vector2d expectedPixel;

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
			 * @param _frustumParameters The [near, far] plane distances of the view frustum.
			 * @param _intrinsics The [sensorWidth, aspectRatio, focalLength] of the pinhole camera model.
			 * @param _imageSize The [width, height] of the image.
			 */
			CorrespondenceResidualBase(Eigen::Vector2d _expectedPixel,
									   Eigen::Vector2d _frustumParameters,
									   Eigen::Vector3d _intrinsics,
									   Eigen::Vector2d _imageSize);

			/**
			 * @destructor
			 */
			virtual ~CorrespondenceResidualBase() = default;
		};

		/**
		 * Represents the residual block when calculating the reprojection error of a known world position to a pixel.
		 */
		class PointCorrespondenceResidual : public CorrespondenceResidualBase {
		private:

			/**
			 * The given world position that results in the expected pixel.
			 */
			Eigen::Vector3d worldPosition;

		public:
			/**
			 * @constructor
			 *
			 * @param _expectedPixel The expected [u, v] pixel location.
			 * @param _worldPosition The given [x, y, z] location in world space.
			 * @param _frustumParameters The [near, far] plane distances of the view frustum.
			 * @param _intrinsics The [sensorWidth, aspectRatio, focalLength] of the pinhole camera model.
			 * @param _imageSize The [width, height] of the image.
			 */
			PointCorrespondenceResidual(Eigen::Vector2d _expectedPixel, Eigen::Vector3d _worldPosition,
										Eigen::Vector2d _frustumParameters, Eigen::Vector3d _intrinsics,
										Eigen::Vector2d _imageSize);

			/**
			 * @destructor
			 */
			~PointCorrespondenceResidual() override = default;

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

		/**
		 * Represents the residual block when calculating the reprojection error of a point on a line in world position
		 * to a pixel.
		 */
		class LineCorrespondenceResidual : public CorrespondenceResidualBase {
		private:

			/**
			 * The [x, y, z] origin in world position of the line that results in the expected pixel.
			 */
			Eigen::Vector3d lineOrigin;

			/**
			 * The [x, y, z] heading in world coordinates of the line that results in the expected pixel.
			 */
			Eigen::Vector3d lineHeading;

		public:
			/**
			 * @constructor
			 *
			 * @param _expectedPixel The expected pixel location.
			 * @param _lineOrigin The [x, y, z] origin in world position of the line that results in the expected pixel.
			 * @param _lineHeading The [x, y, z] heading in world coordinates of the line that results in the expected pixel.
			 * @param _frustumParameters The [near, far] plane distances of the view frustum.
			 * @param _intrinsics The [sensorWidth, aspectRatio, focalLength] of the pinhole camera model.
			 * @param _imageSize The [width, height] of the image.
			 */
			LineCorrespondenceResidual(Eigen::Vector2d _expectedPixel,
									   Eigen::Vector3d _lineOrigin,
									   const Eigen::Vector3d &_lineHeading,
									   Eigen::Vector2d _frustumParameters,
									   Eigen::Vector3d _intrinsics,
									   Eigen::Vector2d _imageSize);

			/**
			 * @destructor
			 */
			~LineCorrespondenceResidual() override = default;

			/**
			 * Calculates the residual error after transforming the world position to a pixel.
			 *
			 * @tparam T Template parameter expected from the ceres-solver.
			 * @param _translation The [x, y, z] translation of the camera in world space for which we optimize.
			 * @param _rotation The [x, y, z] euler angle rotation of the camera around the world axis for which we optimize.
			 * @param _lambda The [l] distance of the point in heading direction from the origin.
			 * @param residual The [u, v] pixel error between the expected and calculated pixel.
			 * @return true
			 */
			template<typename T>
			bool operator()(const T *_translation, const T *_rotation, const T *_lambda, T *residual) const;

			/**
			 * Factory method to hide the residual creation.
			 */
			static ceres::CostFunction *Create(Eigen::Vector2d _expectedPixel,
											   Eigen::Vector3d _lineOrigin,
											   Eigen::Vector3d _lineHeading,
											   Eigen::Vector2d _frustumParameters,
											   Eigen::Vector3d _intrinsics,
											   Eigen::Vector2d _imageSize);

		};

		/**
		 * Represents the residual block when calculating the reprojection error of a point on a plane in world position
		 * to a pixel.
		 */
		class PlaneCorrespondenceResidual : public CorrespondenceResidualBase {
		private:

			/**
			 * The [x, y, z] origin in world position of the line that results in the expected pixel.
			 */
			Eigen::Vector3d planeOrigin;

			/**
			 * The [x, y, z] heading in world coordinates of one side of the plane that results in the expected pixel.
			 */
			Eigen::Vector3d planeSideA;

			/**
			 * The [x, y, z] heading in world coordinates of another side of the plane that results in the expected
			 * pixel.
			 */
			Eigen::Vector3d planeSideB;

		public:
			/**
			 * @constructor
			 *
			 * @param _expectedPixel The expected pixel location.
			 * @param _planeOrigin The [x, y, z] origin in world position of the plane that results in the expected
			 * pixel.
			 * @param _planeSideA The [x, y, z] heading in world coordinates of one side of the plane that results in
			 * the expected pixel.
			 * @param _planeSideB The [x, y, z] heading in world coordinates of another side of the plane that
			 * results in the expected pixel.
			 * @param _frustumParameters The [near, far] plane distances of the view frustum.
			 * @param _intrinsics The [sensorWidth, aspectRatio, focalLength] of the pinhole camera model.
			 * @param _imageSize The [width, height] of the image.
			 */
			PlaneCorrespondenceResidual(Eigen::Vector2d _expectedPixel,
										Eigen::Vector3d _planeOrigin,
										const Eigen::Vector3d &_planeSideA,
										const Eigen::Vector3d &_planeSideB,
										Eigen::Vector2d _frustumParameters,
										Eigen::Vector3d _intrinsics,
										Eigen::Vector2d _imageSize
			);

			/**
			 * @destructor
			 */
			~PlaneCorrespondenceResidual() override = default;

			/**
			 * Calculates the residual error after transforming the world position to a pixel.
			 *
			 * @tparam T Template parameter expected from the ceres-solver.
			 * @param _translation The [x, y, z] translation of the camera in world space for which we optimize.
			 * @param _rotation The [x, y, z] euler angle rotation of the camera around the world axis for which we optimize.
			 * @param _lambda The [l] distance of the point in the direction of one side of the plane from the origin.
			 * @param _mu The [m] distance of the point in the direction of another side of the plane from the origin.
			 * @param residual The [u, v] pixel error between the expected and calculated pixel.
			 * @return true
			 */
			template<typename T>
			bool operator()(const T *_translation, const T *_rotation, const T *_lambda, const T *_mu, T *residual)
			const;

			/**
			 * Factory method to hide the residual creation.
			 */
			static ceres::CostFunction *Create(Eigen::Vector2d _expectedPixel,
											   Eigen::Vector3d _planeOrigin,
											   Eigen::Vector3d _planeSideA,
											   Eigen::Vector3d _planeSideB,
											   Eigen::Vector2d _frustumParameters,
											   Eigen::Vector3d _intrinsics,
											   Eigen::Vector2d _imageSize);

		};

	}
}

#endif //CAMERASTABILIZATION_RESIDUALS_HPP
