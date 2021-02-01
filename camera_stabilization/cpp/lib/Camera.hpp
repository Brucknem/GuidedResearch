//
// Created by brucknem on 27.01.21.
//

#ifndef CAMERASTABILIZATION_CAMERA_HPP
#define CAMERASTABILIZATION_CAMERA_HPP

#include <memory>

#include "Eigen/Dense"
#include "Intrinsics.hpp"
#include "PerspectiveProjection.hpp"

namespace providentia {
	namespace camera {

		/**
		 *	A virtual camera.
		 */
		class Camera {
		private:
//			/**
//			 * The camera intrinsic parameters.
//			 */
//			providentia::camera::Intrinsics intrinsics = providentia::camera::Intrinsics(0, 0, 0, 0);

			/**
			 * Buffer for the calculation of the resulting pixel.
			 */
			Eigen::Vector2f pointInImageSpace;

			/**
			 * Buffer for the intermediate vector in camera space.
			 */
			Eigen::Vector4f pointInCameraSpace;

			/**
			 * The perspective transformation from camera space to normalized device coordinates.
			 */
			std::shared_ptr<providentia::camera::PerspectiveProjection> perspectiveProjection;

			/**
			 * The image size in pixels.
			 */
			Eigen::Vector2f imageSize;

			/**
			 * The camera translation in world space.
			 */
			Eigen::Matrix4f translation;

			/**
			 * The camera rotation in world space.
			 */
			Eigen::Matrix4f rotation, rotationCalculationBuffer;

			/**
			 * The world to camera transformation matrix.
			 */
			Eigen::Matrix4f worldToCamera, cameraToWorld;

			/**
			 * Updates the world to camera transformation based on the translation and rotation.
			 */
			void updateWorldToCamera();

		public:

			/**
			 * @constructor
			 *
			 * @param sensorWidth The physical width [mm] of the camera sensor. Given by the camera specifications.
			 * @param aspectRatio The ratio of width to height of the camera sensor. Given by the camera specifications.
			 * @param imageSize The size [px] of the output image.
			 * @param focalLength The focal length [mm] of the camera. Given by the camera specifications.
			 * @param translation The translation of the camera in world space.
			 * @param rotation The rotation of the camera in world space.
			 */
			Camera(float sensorWidth, float aspectRatio, float focalLength,
				   const Eigen::Vector2f &imageSize, const Eigen::Vector3f &translation = Eigen::Vector3f::Zero(),
				   const Eigen::Vector3f &rotation = Eigen::Vector3f::Zero());

//			/**
//			 * @constructor
//			 *
//			 * @param intrinsics A vector of camera intrinsic parameters.
//			 * @param translation The translation of the camera in world space.
//			 * @param rotation The rotation of the camera in world space.
//			 */
//			Camera(const providentia::camera::Intrinsics &intrinsics, const Eigen::Vector3f &translation,
//				   const Eigen::Vector3f &rotation);

			/**
			 * @destructor
			 */
			virtual ~Camera() = default;

			/**
			 * @get
			 */
			const Eigen::Matrix4f &getTranslation() const;

			/**
			 * @get
			 */
			const Eigen::Matrix4f &getRotation() const;

			/**
			 * @get
			 */
			const Eigen::Matrix4f &getWorldToCameraTransformation() const;

			/**
			 * @get
			 */
			const Eigen::Matrix4f &getCameraToWorldTransformation() const;

			/**
			 * @get
			 */
			const std::shared_ptr<providentia::camera::PerspectiveProjection> &getPerspectiveProjection() const;

			/**
			 * @set
			 */
			void setTranslation(const Eigen::Vector3f &_translation);

			/**
			 * @set
			 */
			void setTranslation(float x, float y, float z);

			/**
			 * @set
			 */
			void setRotation(const Eigen::Vector3f &_rotation);

			/**
			 * @set
			 */
			void setRotation(float x, float y, float z);

			// TODO Document!!
			/**
			 *
			 */
			Eigen::Vector2f operator*(const Eigen::Vector4f &vector);

			/**
			 * Writes the camera object to the stream.
			 */
			friend std::ostream &operator<<(std::ostream &os, const Camera &obj);

		};

		/**
		 * The virtual camera used in the blender test setup.
		 */
		class BlenderCamera : public Camera {
		public:

			/**
			 * @constructor
			 *
			 * @param translation The translation of the camera in world space.
			 * @param rotation The rotation of the camera in world space.
			 */
			explicit BlenderCamera(const Eigen::Vector3f &translation = {0, -10, 5},
								   const Eigen::Vector3f &rotation = {76.5, 0, 0});

			/**
			 * @destructor
			 */
			virtual ~BlenderCamera() =
			default;
		};
	}// namespace camera
}// namespace providentia

#endif//CAMERASTABILIZATION_CAMERA_HPP
