//
// Created by brucknem on 27.01.21.
//

#ifndef CAMERASTABILIZATION_CAMERA_HPP
#define CAMERASTABILIZATION_CAMERA_HPP

#include "Eigen/Dense"
#include "Intrinsics.hpp"
#include <memory>

namespace providentia {
	namespace camera {

		/**
		 *	A virtual camera.
		 */
		class Camera {
		private:
			/**
			 * The camera intrinsic parameters.
			 */
			providentia::camera::Intrinsics cameraMatrix = providentia::camera::Intrinsics(0, 0, 0, 0);

			/**
			 * The camera translation in world space.
			 */
			Eigen::Matrix4f translation;

			/**
			 * The camera rotation in world space.
			 */
			Eigen::Matrix4f rotation, rotationCalculationBuffer;

			/**
			 * The camera view matrix.
			 */
			Eigen::Matrix4f viewMatrix, viewMatrixInverse;

			/**
			 * Updates the view matrix based on the translation and rotation.
			 */
			void setViewMatrix();

		public:

			/**
			 * @constructor
			 *
			 * @param intrinsics A vector of camera intrinsic parameters.
			 * @param translation The translation of the camera in world space.
			 * @param rotation The rotation of the camera in world space.
			 */
			Camera(const Eigen::Vector4f &intrinsics, const Eigen::Vector3f &translation,
				   const Eigen::Vector3f &rotation);

			/**
			 * @constructor
			 *
			 * @param intrinsics A vector of camera intrinsic parameters.
			 * @param translation The translation of the camera in world space.
			 * @param rotation The rotation of the camera in world space.
			 */
			Camera(const providentia::camera::Intrinsics &intrinsics, const Eigen::Vector3f &translation,
				   const Eigen::Vector3f &rotation);

			/**
			 * @destructor
			 */
			virtual ~Camera() = default;

			/**
			 * @get
			 */
			const Intrinsics &getCameraMatrix() const;

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
			const Eigen::Matrix4f &getViewMatrix() const;

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

			/**
			 * @operator
			 */
			Eigen::Vector3f operator*(const Eigen::Vector4f &vector);
		};

		/**
		 * Writes the camera object to the stream.
		 */
		std::ostream &operator<<(std::ostream &os, const Camera &obj);

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
			explicit BlenderCamera(const Eigen::Vector3f &translation = Eigen::Vector3f(0, -10, 5),
								   const Eigen::Vector3f &rotation = Eigen::Vector3f(76.5, 0, 0));

			/**
			 * @destructor
			 */
			virtual ~BlenderCamera() = default;
		};
	}// namespace camera
}// namespace providentia

#endif//CAMERASTABILIZATION_CAMERA_HPP
