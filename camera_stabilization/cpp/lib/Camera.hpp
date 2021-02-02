//
// Created by brucknem on 27.01.21.
//

#ifndef CAMERASTABILIZATION_CAMERA_HPP
#define CAMERASTABILIZATION_CAMERA_HPP

#include <memory>

#include "opencv2/opencv.hpp"
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
			 * The currently rendering image.
			 */
			cv::Mat imageBuffer;

			/**
			 * Buffer for the calculation of the resulting pixel.
			 */
			Eigen::Vector2f pointInImageSpace;

			/**
			 * The perspective transformation from camera space to normalized device coordinates.
			 */
			std::shared_ptr<providentia::camera::PerspectiveProjection> perspectiveProjection;

			/**
			 * The image size in pixels.
			 */
			Eigen::Vector2i imageSize;

			/**
			 * The camera translation in world space.
			 */
			Eigen::Vector4f translation;

			/**
			 * The camera rotation in world space.
			 */
			Eigen::Vector3f rotation;


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
				   const Eigen::Vector2i &imageSize, const Eigen::Vector3f &translation = Eigen::Vector3f::Zero(),
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
			const std::shared_ptr<providentia::camera::PerspectiveProjection> &getPerspectiveProjection() const;

			/**
			 * @get The rotation matrix from the angle axis rotation.
			 */
			Eigen::Matrix4f getRotationMatrix() const;

			/**
			 * @get
			 */
			cv::Mat getImage() const;

			/**
			 * @set
			 */
			void setTranslation(const Eigen::Vector3f &_translation);

			/**
			 * @set
			 */
			void setRotation(const Eigen::Vector3f &_rotation);

			/**
			 * @set
			 */
			void setRotation(float x, float y, float z);

			/**
			 * Performs the whole rendering pipeline. <br>
			 *
			 * World -> Camera Space <br>
			 * Camera Space -> Frustum <br>
			 * Frustum -> Clip Space <br>
			 * Clip Space -> Normalized device coordinates <br>
			 * Normalized device coordinates -> Pixels <br>
			 */
			Eigen::Vector2f operator*(const Eigen::Vector4f &vector);

			/**
			 * Transforms a given vector from camera to world space.
			 */
			Eigen::Vector4f toWorldSpace(const Eigen::Vector4f &vector);

			/**
			 * Transforms a given vector from world to camera space.
			 */
			Eigen::Vector4f toCameraSpace(const Eigen::Vector4f &vector);

			/**
			 * Writes the camera object to the stream.
			 */
			friend std::ostream &operator<<(std::ostream &os, const Camera &obj);

			/**
			 * Renders the given vector onto the virtual image.
			 */
			void render(const Eigen::Vector4f &vector, const cv::Vec3f &color = {1, 1, 1});

			/**
			 * Renders the given vector onto the virtual image.
			 */
			void render(float x, float y, float z, const cv::Vec3f &color = {1, 1, 1});

			/**
			 * Clears the image buffer.
			 */
			void resetImage();
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
								   const Eigen::Vector3f &rotation = {90, 0, 0});

			/**
			 * @destructor
			 */
			~BlenderCamera() override = default;
		};
	}// namespace camera
}// namespace providentia

#endif//CAMERASTABILIZATION_CAMERA_HPP
