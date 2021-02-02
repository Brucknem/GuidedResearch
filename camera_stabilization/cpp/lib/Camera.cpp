//
// Created by brucknem on 27.01.21.
//

#include "Camera.hpp"
#include <iostream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"

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

		Eigen::Vector2f Camera::operator*(const Eigen::Vector4f &vector) {
			pointInImageSpace = *perspectiveProjection * toCameraSpace(vector);
			pointInImageSpace = (pointInImageSpace + Eigen::Vector2f::Ones()) / 2;
			pointInImageSpace.x() *= imageSize.x();
			pointInImageSpace.y() *= imageSize.y();
			return pointInImageSpace;
		}

		void Camera::setTranslation(const Eigen::Vector3f &_translation) {
			translation = {_translation.x(), _translation.y(), _translation.z(), 0};
		}

		void Camera::setRotation(const Eigen::Vector3f &_rotation) {
			rotation = _rotation;
		}

		void Camera::setRotation(float x, float y, float z) {
			setRotation({x, y, z});
		}

		std::ostream &operator<<(std::ostream &os, const Camera &obj) {
//			os << "Intrinsics" << std::endl
//			   << obj.intrinsics << std::endl;
			os << "Translation" << std::endl
			   << obj.translation << std::endl;
			os << "Rotation" << std::endl
			   << obj.rotation << std::endl;
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

		Eigen::Matrix4f Camera::getRotationMatrix() const {
			Eigen::Vector3f rotationInRadians = rotation;
			rotationInRadians.x() *= -1;
			rotationInRadians.y() *= -1;
			rotationInRadians *= M_PI / 180;

			Eigen::Matrix4f zAxis;
			float theta = rotationInRadians.z();
			zAxis << cos(theta), -sin(theta), 0, 0,
					sin(theta), cos(theta), 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1;

			Eigen::Matrix4f yAxis;
			theta = rotationInRadians.y();
			yAxis << cos(theta), 0, sin(theta), 0,
					0, 1, 0, 0,
					-sin(theta), 0, cos(theta), 0,
					0, 0, 0, 1;

			Eigen::Matrix4f xAxis;
			theta = rotationInRadians.x();
			xAxis << 1, 0, 0, 0,
					0, cos(theta), -sin(theta), 0,
					0, sin(theta), cos(theta), 0,
					0, 0, 0, 1;

			Eigen::Matrix4f initialCameraRotation;
			initialCameraRotation << 1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, -1, 0,
					0, 0, 0, 1;
			return (initialCameraRotation * zAxis * yAxis * xAxis);
		}

		Eigen::Vector4f Camera::toCameraSpace(const Eigen::Vector4f &vector) {
			return getRotationMatrix().inverse() * (vector - translation);
		}

		Eigen::Vector4f Camera::toWorldSpace(const Eigen::Vector4f &vector) {
			return (getRotationMatrix() * vector) + translation;
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
