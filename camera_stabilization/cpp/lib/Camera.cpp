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

		template<typename T>
		Eigen::Matrix<T, 4, 4> getRotationMatrix(Eigen::Matrix<T, 3, 1> rotation) {
			Eigen::Vector3d rotationInRadians = rotation;
			rotationInRadians.x() *= -1;
			rotationInRadians.y() *= -1;
			rotationInRadians *= M_PI / 180;

			Eigen::Matrix4d zAxis;
			double theta = rotationInRadians.z();
			zAxis << cos(theta), -sin(theta), 0, 0,
					sin(theta), cos(theta), 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1;

			Eigen::Matrix4d yAxis;
			theta = rotationInRadians.y();
			yAxis << cos(theta), 0, sin(theta), 0,
					0, 1, 0, 0,
					-sin(theta), 0, cos(theta), 0,
					0, 0, 0, 1;

			Eigen::Matrix4d xAxis;
			theta = rotationInRadians.x();
			xAxis << 1, 0, 0, 0,
					0, cos(theta), -sin(theta), 0,
					0, sin(theta), cos(theta), 0,
					0, 0, 0, 1;

			Eigen::Matrix4d initialCameraRotation;
			initialCameraRotation << 1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, -1, 0,
					0, 0, 0, 1;
			return (initialCameraRotation * zAxis * yAxis * xAxis);
		}

		Camera::Camera(const double sensorWidth, const double aspectRatio, const double focalLength,
					   const Eigen::Vector2i &_imageSize, const Eigen::Vector3d &translation,
					   const Eigen::Vector3d &rotation) {
			perspectiveProjection = std::make_shared<providentia::camera::PerspectiveProjection>(sensorWidth,
																								 aspectRatio,
																								 focalLength);
			imageSize = _imageSize;
			setTranslation(translation);
			setRotation(rotation);
		}

//		Camera::Camera(const Intrinsics &_intrinsics, const Eigen::Vector3d &translation,
//					   const Eigen::Vector3d &rotation) {
//			intrinsics = providentia::camera::Intrinsics(_intrinsics);
//			setTranslation(translation);
//			setRotation(rotation);
//		}

		Eigen::Vector2d Camera::operator*(const Eigen::Vector4d &vector) {
			pointInImageSpace = *perspectiveProjection * toCameraSpace(vector);
			pointInImageSpace = (pointInImageSpace + Eigen::Vector2d::Ones()) / 2;
			pointInImageSpace.x() *= imageSize.x();
			pointInImageSpace.y() *= imageSize.y();
			return pointInImageSpace;
		}

		void Camera::setTranslation(const Eigen::Vector3d &_translation) {
			translation = {_translation.x(), _translation.y(), _translation.z(), 0};
		}

		void Camera::setRotation(const Eigen::Vector3d &_rotation) {
			rotation = _rotation;
		}

		void Camera::setRotation(double x, double y, double z) {
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


		void Camera::render(double x, double y, double z, const cv::Vec3d &color) {
			render({x, y, z, 1}, color);
		}

		void Camera::render(const Eigen::Vector4d &vector, const cv::Vec3d &color) {
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

					double distance = (nearestPixel.cast<double>() - pointInImageSpace).norm();
					cv::Vec4d _color = {color[0], color[1], color[2], (distance / (double) sqrt(2))};

					imageBuffer.at<cv::Vec4d>(nearestPixel.y(), nearestPixel.x()) = _color;
				}
			}
		}

		void Camera::resetImage() {
			imageBuffer = cv::Mat::zeros(cv::Size(imageSize.x(), imageSize.y()), CV_32FC4);
		}

		cv::Mat Camera::getImage() const {
			return imageBuffer.clone();
		}

		Eigen::Matrix4d Camera::getRotationMatrix() const {
			return providentia::camera::getRotationMatrix<double>(rotation);
		}

		Eigen::Vector4d Camera::toCameraSpace(const Eigen::Vector4d &vector) {
			return getRotationMatrix().inverse() * (vector - translation);
		}

		Eigen::Vector4d Camera::toWorldSpace(const Eigen::Vector4d &vector) {
			return (getRotationMatrix() * vector) + translation;
		}

		const Eigen::Vector4d &Camera::getTranslation() {
			return translation;
		}

		const Eigen::Matrix<double, 3, 1> &Camera::getRotation() {
			return rotation;
		}

#pragma endregion Camera

#pragma region BlenderCamera

		BlenderCamera::BlenderCamera(
				const Eigen::Vector3d &translation,
				const Eigen::Vector3d &rotation) : Camera(
				32, 1920.0f / 1200, 20, {1920, 1200}, translation, rotation) {}

#pragma endregion BlenderCamera

	}// namespace camera
}// namespace providentia
