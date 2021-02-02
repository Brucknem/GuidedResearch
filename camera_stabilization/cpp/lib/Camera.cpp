//
// Created by brucknem on 27.01.21.
//

#include "Camera.hpp"
#include <iostream>

namespace providentia {
	namespace camera {
#pragma region Camera


		Camera::Camera(const float sensorWidth, const float aspectRatio, const float focalLength,
					   const Eigen::Vector2i &_imageSize, const Eigen::Vector3f &translation,
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
			pointInCameraSpace = worldToCamera * vector;
			pointInImageSpace = *perspectiveProjection * pointInCameraSpace;
			pointInImageSpace = (pointInImageSpace + Eigen::Vector2f::Ones()) / 2;
			pointInImageSpace.x() *= imageSize.x();
			pointInImageSpace.y() *= imageSize.y();
			return pointInImageSpace;
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

		const Eigen::Matrix4f &Camera::getCameraToWorldTransformation() const {
			return cameraToWorld;
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

		const std::shared_ptr<providentia::camera::PerspectiveProjection> &Camera::getPerspectiveProjection() const {
			return perspectiveProjection;
		}


		void Camera::render(float x, float y, float z, const cv::Vec3f &color) {
			render({x, y, z, 1}, color);
		}

		void Camera::render(const Eigen::Vector4f &vector, const cv::Vec3f &color) {
			if (imageBuffer.empty()) {
				resetImage();
			}
			*this * vector;

			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < 2; ++j) {
					Eigen::Vector2i nearestPixel = pointInImageSpace.cast<int>();
					nearestPixel.x() += i;
					nearestPixel.y() += j;
					nearestPixel.y() = imageSize.y() - 1 - nearestPixel.y();

					if (nearestPixel.x() >= imageSize.x() || nearestPixel.y() >= imageSize.y() ||
						nearestPixel.x() < 0 || nearestPixel.y() < 0) {
//						std::cout << "[" << nearestPixel[0] << ", " << nearestPixel[1] << "] out of frustum"
//								  << std::endl;
						continue;
					}

					float distance = (nearestPixel.cast<float>() - pointInImageSpace).norm();
					cv::Vec4f _color = {color[0], color[1], color[2], (distance / (float) sqrt(2))};

					imageBuffer.at<cv::Vec4f>(nearestPixel.y(), nearestPixel.x()) = _color;
				}
			}
		}

		void Camera::resetImage() {
			imageBuffer = cv::Mat::zeros(cv::Size(imageSize.x(), imageSize.y()), CV_32FC4);
		}

		cv::Mat Camera::getImage() const {
			return imageBuffer.clone();
		}

#pragma endregion Camera

#pragma region BlenderCamera

		BlenderCamera::BlenderCamera(
				const Eigen::Vector3f &translation,
				const Eigen::Vector3f &rotation) : Camera(
				32, 1920.0f / 1200, 20, {1920, 1200}, translation, rotation) {}

#pragma endregion BlenderCamera

	}// namespace camera
}// namespace providentia
