//
// Created by brucknem on 27.01.21.
//

#include "Camera.hpp"
#include <iostream>

namespace providentia {
	namespace camera {
#pragma region Camera

		Camera::Camera(const Eigen::Vector4f &intrinsics, const Eigen::Vector3f &translation,
					   const Eigen::Vector3f &rotation) : Camera(providentia::camera::Intrinsics(intrinsics),
																 translation, rotation) {}

		Camera::Camera(const Intrinsics &intrinsics, const Eigen::Vector3f &translation,
					   const Eigen::Vector3f &rotation) {
			cameraMatrix = providentia::camera::Intrinsics(intrinsics);
			setTranslation(translation);
			setRotation(rotation);
		}


		const Intrinsics &Camera::getCameraMatrix() const {
			return cameraMatrix;
		}

		const Eigen::Matrix4f &Camera::getTranslation() const {
			return translation;
		}

		const Eigen::Matrix4f &Camera::getRotation() const {
			return rotation;
		}

		Eigen::Vector3f Camera::operator*(const Eigen::Vector4f &vector) {
			Eigen::Vector4f pointInCameraSpace = viewMatrix * vector;
			return cameraMatrix * pointInCameraSpace;
		}

		void Camera::setTranslation(const Eigen::Vector3f &_translation) {
			setTranslation(_translation(0), _translation(1), _translation(2));
		}

		void Camera::setTranslation(float x, float y, float z) {
			translation = Eigen::Matrix4f::Identity();
			translation(0, 3) = x;
			translation(1, 3) = y;
			translation(2, 3) = z;

			setViewMatrix();
		}

		void Camera::setRotation(const Eigen::Vector3f &_rotation) {
			setRotation(_rotation(0), _rotation(1), _rotation(2));
		}

		void Camera::setRotation(float x, float y, float z) {
			rotation = Eigen::Matrix4f::Identity();
			rotation(2, 2) = -1;

			rotationCalculationBuffer = Eigen::Matrix4f::Identity();

			rotationCalculationBuffer.block<3, 3>(0, 0) = Eigen::AngleAxisf(z * M_PI / 180,
																			Eigen::Vector3f::UnitZ()).matrix();
			rotation = rotation * rotationCalculationBuffer;

			rotationCalculationBuffer.block<3, 3>(0, 0) = Eigen::AngleAxisf(-y * M_PI / 180,
																			Eigen::Vector3f::UnitY()).matrix();
			rotation = rotation * rotationCalculationBuffer;

			rotationCalculationBuffer.block<3, 3>(0, 0) = Eigen::AngleAxisf(-x * M_PI / 180,
																			Eigen::Vector3f::UnitX()).matrix();
			rotation = rotation * rotationCalculationBuffer;

			setViewMatrix();
		}

		const Eigen::Matrix4f &Camera::getViewMatrix() const {
			return viewMatrix;
		}

		void Camera::setViewMatrix() {
			viewMatrix = rotation * translation;
			viewMatrixInverse = viewMatrix.inverse();
		}

		std::ostream &operator<<(std::ostream &os, const Camera &obj) {
			os << "Intrinsics" << std::endl
			   << obj.getCameraMatrix() << std::endl;
			os << "Translation" << std::endl
			   << obj.getTranslation() << std::endl;
			os << "Rotation" << std::endl
			   << obj.getRotation() << std::endl;
			os << "View" << std::endl
			   << obj.getViewMatrix();
			return os;
		}

#pragma endregion Camera

#pragma region BlenderCamera

		BlenderCamera::BlenderCamera(const Eigen::Vector3f &translation, const Eigen::Vector3f &rotation) : Camera(
				providentia::camera::BlenderIntrinsics(), translation, rotation) {}

#pragma endregion BlenderCamera

	}// namespace camera
}// namespace providentia
