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
		 * @get The rotation matrix from the angle axis rotation.
		 */
		template<typename T>
		Eigen::Matrix<T, 4, 4> getRotationMatrix(Eigen::Matrix<T, 3, 1> rotation);

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
			Eigen::Vector2d pointInImageSpace;

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
			Eigen::Vector4d translation;

			/**
			 * The camera rotation in world space.
			 */
			Eigen::Matrix<double, 3, 1> rotation;

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
			Camera(double sensorWidth, double aspectRatio, double focalLength,
				   const Eigen::Vector2i &imageSize, const Eigen::Vector3d &translation = Eigen::Vector3d::Zero(),
				   const Eigen::Vector3d &rotation = Eigen::Vector3d::Zero());

//			/**
//			 * @constructor
//			 *
//			 * @param intrinsics A vector of camera intrinsic parameters.
//			 * @param translation The translation of the camera in world space.
//			 * @param rotation The rotation of the camera in world space.
//			 */
//			Camera(const providentia::camera::Intrinsics &intrinsics, const Eigen::Vector3d &translation,
//				   const Eigen::Vector3d &rotation);

			/**
			 * @destructor
			 */
			virtual ~Camera() = default;

			/**
			 * @get
			 */
			const std::shared_ptr<providentia::camera::PerspectiveProjection> &getPerspectiveProjection() const;

			/**
			 * @get
			 */
			cv::Mat getImage() const;

			/**
			 * @get The rotation matrix from the angle axis rotation.
			 */
			Eigen::Matrix4d getRotationMatrix() const;

			/**
			 * @set
			 */
			void setTranslation(const Eigen::Vector3d &_translation);

			/**
			 * @set
			 */
			void setRotation(const Eigen::Vector3d &_rotation);

			/**
			 * @set
			 */
			void setRotation(double x, double y, double z);

			/**
			 * Performs the whole rendering pipeline. <br>
			 *
			 * World -> Camera Space <br>
			 * Camera Space -> Frustum <br>
			 * Frustum -> Clip Space <br>
			 * Clip Space -> Normalized device coordinates <br>
			 * Normalized device coordinates -> Pixels <br>
			 */
			Eigen::Vector2d operator*(const Eigen::Vector4d &vector);

			/**
			 * Transforms a given vector from camera to world space.
			 */
			Eigen::Vector4d toWorldSpace(const Eigen::Vector4d &vector);

			/**
			 * Transforms a given vector from world to camera space.
			 */
			Eigen::Vector4d toCameraSpace(const Eigen::Vector4d &vector);

			/**
			 * Writes the camera object to the stream.
			 */
			friend std::ostream &operator<<(std::ostream &os, const Camera &obj);

			/**
			 * Renders the given vector onto the virtual image.
			 */
			void render(const Eigen::Vector4d &vector, const cv::Vec3d &color = {1, 1, 1});

			/**
			 * Renders the given vector onto the virtual image.
			 */
			void render(double x, double y, double z, const cv::Vec3d &color = {1, 1, 1});

			/**
			 * Clears the image buffer.
			 */
			void resetImage();

			const Eigen::Vector4d &getTranslation();

			const Eigen::Matrix<double, 3, 1> &getRotation();
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
			explicit BlenderCamera(const Eigen::Vector3d &translation = {0, -10, 5},
								   const Eigen::Vector3d &rotation = {90, 0, 0});

			/**
			 * @destructor
			 */
			~BlenderCamera() override = default;
		};
	}// namespace camera
}// namespace providentia

#endif//CAMERASTABILIZATION_CAMERA_HPP
