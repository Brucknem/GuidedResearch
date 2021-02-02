//
// Created by brucknem on 28.01.21.
//

#ifndef CAMERASTABILIZATION_PERSPECTIVEPROJECTION_HPP
#define CAMERASTABILIZATION_PERSPECTIVEPROJECTION_HPP

#include <ostream>
#include "Eigen/Dense"


namespace providentia {
	namespace camera {

		/**
		 * Class performing the perspective transformation from camera space to normalized device coordinates.
		 */
		class PerspectiveProjection {
		private:
			/**
			 * The frustum matrix according to the view frustum of the pinhole camera model.
			 */
			Eigen::Matrix4d frustum;

			/**
			 * The matrix mapping the points in the view frustum to normalized device coordinates.
			 */
			Eigen::Matrix4d normalization;

			/**
			 * The total projection matrix mapping from camera space to image space.
			 */
			Eigen::Matrix4d projection;

			/**
			 * The field of view angles (rad) of the camera related to the focal length and sensor width.
			 */
			Eigen::Vector2d fieldOfView;

			/**
			 * The top right and lower left points on the near image plane.
			 */
			Eigen::Vector2d topRight, lowerLeft;

			/**
			 * The distances [m] of the near and far planes of the view frustum.
			 * frustumPlaneDistances.x() == near plane distance
			 * frustumPlaneDistances.y() == far plane distance
			 */
			Eigen::Vector2d frustumPlaneDistances;

			/**
			 * Updates the frustum, normalization and total projection matrix based on the current camera settings.
			 */
			void updateMatrices();

		public:
			/**
			 * @constructor
			 *
			 * @param sensorWidth The physical width [mm] of the camera sensor. Given by the camera specifications.
			 * @param aspectRatio The ratio of width to height of the camera sensor. Given by the camera specifications.
			 * @param focalLength The focal length [mm] of the camera. Given by the camera specifications.
			 * @param nearPlaneDistance The distance [m] of the near plane of the view frustum.
			 * @param farPlaneDistance The distance [m] of the far plane of the view frustum.
			 */
			PerspectiveProjection(double sensorWidth, double aspectRatio, double focalLength,
								  double nearPlaneDistance = 1.,
								  double farPlaneDistance = 1000.);

			/**
			 * @destructor
			 */
			virtual ~PerspectiveProjection() = default;

			/**
			 * Transforms the given vector to the frustum.
			 *
			 * @param vectorInCameraSpace The vector given in camera space.
			 * @return The transformed vector in the frustum. w coordinate is normalized to 1.
			 */
			Eigen::Vector4d toFrustum(const Eigen::Vector4d &vectorInCameraSpace);

			/**
			 * Transforms the given vector to clip space.
			 *
			 * @param vectorInCameraSpace The vector given in camera space.
			 * @return The transformed vector in the clip space. <br>
			 * 			x, y, z are in range [-1, 1]
			 */
			Eigen::Vector3d toClipSpace(const Eigen::Vector4d &vectorInCameraSpace);

			/**
			 * Projects the given vector from camera space to normalized device coordinates.
			 *
			 * @param vectorInCameraSpace The vector given in camera space.
			 * @return The transformed vector in the normalized device coordinates.
			 */
			Eigen::Vector2d operator*(const Eigen::Vector4d &vectorInCameraSpace);

			/**
			 * @stream
			 */
			friend std::ostream &operator<<(std::ostream &os, const PerspectiveProjection &perspective);

			/**
			 * Sets the field of view based on the physical camera parameters.
			 *
			 * @param sensorWidth The physical width [mm] of the camera sensor. Given by the camera specifications.
			 * @param aspectRatio The ratio of width to height of the camera sensor. Given by the camera specifications.
			 * @param focalLength The focal length [mm] of the camera. Given by the camera specifications.
			 */
			void setFieldOfView(double sensorWidth, double aspectRatio, double focalLength);

			/**
			 * Sets the size of the image plane based on the field of view, the aspect ratio of the sensor and the distance of the near plane.
			 *
			 * @param nearPlaneDistance The distance [m] of the near plane of the view frustum.
			 * @param aspectRatio The ratio of width to height of the camera sensor. Given by the camera specifications.
			 */
			void setImagePlaneBounds(double nearPlaneDistance, double aspectRatio);

			/**
			 * Sets the near and far plane distance of the frustum
			 *
			 * @param nearPlaneDistance The distance [m] of the near plane of the view frustum.
			 * @param farPlaneDistance The distance [m] of the far plane of the view frustum.
			 */
			void setFrustumPlaneDistances(double nearPlaneDistance, double farPlaneDistance);

			/**
			 * @get
			 */
			const Eigen::Vector2d &getTopRight() const;

			/**
			 * @get
			 */
			const Eigen::Vector2d &getLowerLeft() const;

			/**
			 * @get
			 */
			const Eigen::Vector2d &getFrustumPlaneDistances() const;
		};

	}
}
#endif //CAMERASTABILIZATION_PERSPECTIVEPROJECTION_HPP
