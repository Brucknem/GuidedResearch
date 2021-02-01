//
// Created by brucknem on 27.01.21.
//

#include "Camera.hpp"
#include <iostream>

namespace providentia {
	namespace camera {
#pragma region Camera


		Camera::Camera(const float sensorWidth, const float aspectRatio, const float focalLength,
					   const Eigen::Vector2f &_imageSize, const Eigen::Vector3f &translation,
					   const Eigen::Vector3f &rotation) {
			perspectiveProjection = std::make_shared<providentia::camera::PerspectiveProjection>(sensorWidth,
																								 aspectRatio,
																								 focalLength);
			imageSize = _imageSize;
			setTranslation(translation);
			setRotation(rotation);
		}

//		Camera::Camera(const Intrinsics &_intrinsics, const Eigen::Vector3f &translation,
//					   const Eigen::Vector3f &rotation) {
//			intrinsics = providentia::camera::Intrinsics(_intrinsics);
//			setTranslation(translation);
//			setRotation(rotation);
//		}

		const Eigen::Matrix4f &Camera::getTranslation() const {
			return translation;
		}

		const Eigen::Matrix4f &Camera::getRotation() const {
			return rotation;
		}

		Eigen::Vector2f Camera::operator*(const Eigen::Vector4f &vector) {
			Eigen::Vector4f pointInCameraSpace = worldToCamera * vector;
			Eigen::Vector2f normalizedDeviceCoordinate = *perspectiveProjection * pointInCameraSpace;
			return (normalizedDeviceCoordinate + Eigen::Vector2f::Ones()) / 2;
		}

		void Camera::setTranslation(const Eigen::Vector3f &_translation) {
			setTranslation(_translation(0), _translation(1), _translation(2));
		}

		void Camera::setTranslation(float x, float y, float z) {
			translation = Eigen::Matrix4f::Identity();
			translation(0, 3) = x;
			translation(1, 3) = y;
			translation(2, 3) = z;

			updateWorldToCamera();
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

			updateWorldToCamera();
		}

		const Eigen::Matrix4f &Camera::getWorldToCameraTransformation() const {
			return worldToCamera;
		}

		void Camera::updateWorldToCamera() {
			cameraToWorld = rotation;
			cameraToWorld.block<3, 1>(0, 3) = translation.block<3, 1>(0, 3);
			worldToCamera = cameraToWorld.inverse();
		}

		std::ostream &operator<<(std::ostream &os, const Camera &obj) {
//			os << "Intrinsics" << std::endl
//			   << obj.intrinsics << std::endl;
			os << "Translation" << std::endl
			   << obj.translation << std::endl;
			os << "Rotation" << std::endl
			   << obj.rotation << std::endl;
			os << "View" << std::endl
			   << obj.worldToCamera;
			return os;
		}

#pragma endregion Camera

#pragma region BlenderCamera

		BlenderCamera::BlenderCamera(const Eigen::Vector3f &translation, const Eigen::Vector3f &rotation) : Camera(
				8, 1920.0f / 1200, 4, Eigen::Vector2f(1920, 1200), translation, rotation) {}

#pragma endregion BlenderCamera

	}// namespace camera
}// namespace providentia
