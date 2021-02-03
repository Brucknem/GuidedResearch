//
// Created by brucknem on 02.02.21.
//

#ifndef CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP
#define CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP

#include <utility>

#include "ceres/ceres.h"
#include "opencv2/opencv.hpp"

namespace providentia {
	namespace calibration {

		template<typename T>
		Eigen::Matrix<T, 4, 4> getRotationMatrix(const T *_rotation) {
			Eigen::Matrix<T, 3, 1> rotationInRadians{-_rotation[0], -_rotation[1], _rotation[2]};
			rotationInRadians *= (T) (M_PI / 180);

			T zero = (T) 0;
			T one = (T) 1;

			Eigen::Matrix<T, 4, 4> zAxis;
			T theta = rotationInRadians.z();
			zAxis << cos(theta), -sin(theta), zero, zero,
					sin(theta), cos(theta), zero, zero,
					zero, zero, one, zero,
					zero, zero, zero, one;

			Eigen::Matrix<T, 4, 4> yAxis;
			theta = rotationInRadians.y();
			yAxis << cos(theta), zero, sin(theta), zero,
					zero, one, zero, zero,
					-sin(theta), zero, cos(theta), zero,
					zero, zero, zero, one;

			Eigen::Matrix<T, 4, 4> xAxis;
			theta = rotationInRadians.x();
			xAxis << one, zero, zero, zero,
					zero, cos(theta), -sin(theta), zero,
					zero, sin(theta), cos(theta), zero,
					zero, zero, zero, one;

			Eigen::Matrix<T, 4, 4> rotation;
			rotation << one, zero, zero, zero,
					zero, one, zero, zero,
					zero, zero, -one, zero,
					zero, zero, zero, one;

			return (rotation * zAxis * yAxis * xAxis);
		}

		template<typename T>
		Eigen::Matrix<T, 4, 1> toCameraSpace(const T *_translation, const T *_rotation, const T *vector) {
			// TODO Remove inverse
			return getRotationMatrix<T>(_rotation).inverse() *
				   (Eigen::Matrix<T, 4, 1>(
						   vector[0] - _translation[0],
						   vector[1] - _translation[1],
						   vector[2] - _translation[2],
						   (T) 1));
		}

		template<typename T>
		Eigen::Matrix<T, 4, 4> getFrustum(const T *_frustumParameters) {
			T zero = (T) 0;
			T near = _frustumParameters[0];
			T far = _frustumParameters[1];

			Eigen::Matrix<T, 4, 4> frustum;
			frustum << near, zero, zero, zero,
					zero, near, zero, zero,
					zero, zero, near + far, -near * far,
					zero, zero, (T) 1, zero;
			return frustum;
		}

		template<typename T>
		Eigen::Matrix<T, 4, 1> normalize(const Eigen::Matrix<T, 4, 1> &vector) {
			return vector / vector[3];
		}

		template<typename T>
		Eigen::Matrix<T, 4, 1> toFrustum(const T *_frustumParameters, const T *vector) {
			Eigen::Matrix<T, 4, 1> pointInFrustum = getFrustum(_frustumParameters) *
													Eigen::Matrix<T, 4, 1>{vector[0], vector[1], vector[2], vector[3]};
			return providentia::calibration::normalize(pointInFrustum);
		}


		template<typename T>
		Eigen::Matrix<T, 4, 4>
		getClipSpace(const T *_frustumParameters, const T *_intrinsics) {
			T zero = (T) 0;
			T one = (T) 1;

			T near = _frustumParameters[0];
			T far = _frustumParameters[1];

			T fieldOfViewX = (T) (2.) * atan((T) (0.5) * (_intrinsics[0] / _intrinsics[2]));
			T right = near * tan(fieldOfViewX * (T) (0.5));
			T top = right / _intrinsics[1];

			Eigen::Matrix<T, 4, 4> normalization;
			normalization <<
						  (T) (2) / (right - -right), zero, zero, -(right + -right) / (right - -right),
					zero, (T) (2) / (top - -top), zero, -(top + -top) / (top - -top),
					zero, zero, (T) (2) / (far - near), -(far + near) / (far - near),
					zero, zero, zero, one;

			return normalization * getFrustum(_frustumParameters);
		}

		template<typename T>
		Eigen::Matrix<T, 4, 1>
		toClipSpace(const T *_frustumParameters, const T *_intrinsics, const T *vector) {
			Eigen::Matrix<T, 4, 1> pointInClipSpace = getClipSpace(_frustumParameters, _intrinsics) *
													  Eigen::Matrix<T, 4, 1>{vector[0], vector[1], vector[2],
																			 vector[3]};
			return providentia::calibration::normalize(pointInClipSpace);
		}


		template<typename T>
		Eigen::Matrix<T, 2, 1> toNormalizedDeviceCoordinates(const T *vector) {
			return Eigen::Matrix<T, 2, 1>(vector[0], vector[1]);
		}

		template<typename T>
		Eigen::Matrix<T, 2, 1> toImageSpace(const T *_imageSize, const T *vector) {
			Eigen::Matrix<T, 2, 1> pixel{vector[0], vector[1]};
			pixel += Eigen::Matrix<T, 2, 1>(1, 1);
			pixel *= (T) 0.5;
			return Eigen::Matrix<T, 2, 1>(pixel[0] * _imageSize[0], pixel[1] * _imageSize[1]);
		}

		template<typename T>
		Eigen::Matrix<T, 2, 1>
		render(const T *_translation, const T *_rotation, const T *_frustumParameters, const T *_intrinsics,
			   const T *_imageSize, const T *vector) {
			Eigen::Matrix<T, 4, 1> point = toCameraSpace(_translation, _rotation, vector);
			point = toClipSpace(_frustumParameters, _intrinsics, point.data());
			Eigen::Matrix<T, 2, 1> pixel = toNormalizedDeviceCoordinates(point.data());
			pixel = toImageSpace(_imageSize, pixel.data());
			return pixel;
		}

		void render(const Eigen::Vector3d &_translation, const Eigen::Vector3d &_rotation,
					const Eigen::Vector2d &_frustumParameters, const Eigen::Vector3d &_intrinsics,
					const Eigen::Vector4d &vector, const cv::Vec3d &color, cv::Mat image) {
			Eigen::Vector2d imageSize(image.cols, image.rows);
			Eigen::Vector2d pointInImageSpace = render(
					_translation.data(), _rotation.data(),
					_frustumParameters.data(), _intrinsics.data(),
					imageSize.data(),
					vector.data());

			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < 2; ++j) {
					Eigen::Vector2i nearestPixel = pointInImageSpace.cast<int>();
					nearestPixel.x() += i;
					nearestPixel.y() += j;
					double distance = (nearestPixel.cast<double>() - pointInImageSpace).norm();
					nearestPixel.y() = imageSize.y() - 1 - nearestPixel.y();

//					std::cout << "[" << nearestPixel.x() << ", " << nearestPixel.y() << "] - " << distance;
					if (nearestPixel.x() >= imageSize.x() || nearestPixel.y() >= imageSize.y() ||
						nearestPixel.x() < 0 || nearestPixel.y() < 0) {
//						std::cout << " out of frustum";
//						std::cout << std::endl;
						continue;
					}
//					std::cout << std::endl;

					cv::Vec4d _color = {color[0], color[1], color[2], (distance / (double) sqrt(2))};

					// 1200, 1920
					image.at<cv::Vec4d>(nearestPixel.y(), nearestPixel.x()) = _color;
				}
			}
		}

		struct ReprojectionResidual {
		private:
			const Eigen::Vector2d pixel;
			const Eigen::Vector3d worldCoordinate;

			Eigen::Vector2d frustumParameters{1, 1000};
			Eigen::Vector3d intrinsics{8, 1920. / 1200., 4};


		public:
			ReprojectionResidual(Eigen::Vector2d _pixel, Eigen::Vector3d _worldCoordinate) :
					pixel(std::move(_pixel)), worldCoordinate(std::move(_worldCoordinate)) {}

			template<typename T>
			bool operator()(const T *_translation, const T *_rotation, T *residual) const {
				// TODO World Space -> Camera Space

				Eigen::Matrix<T, 4, 1> point{(T) worldCoordinate.x(), (T) worldCoordinate.y(),
											 (T) worldCoordinate.z(),
											 (T) 1};
				Eigen::Matrix<T, 4, 1> translation{_translation[0], _translation[1], _translation[2], (T) 0};

				std::cout << point << std::endl;

				point = toCameraSpace(_translation, _rotation, point.data());
				std::cout << point << std::endl;

				// TODO Camera Space -> Clip Space
				Eigen::Matrix<T, 2, 1> _frustumParameters{(T) frustumParameters.x(), (T) frustumParameters.y()};
				Eigen::Matrix<T, 3, 1> _intrinsics{(T) intrinsics.x(), (T) intrinsics.y(), (T) intrinsics.z()};
				point = toClipSpace(_frustumParameters.data(), _intrinsics.data(), point.data());
				std::cout << point << std::endl;

				// TODO Clip Space -> Normalized Device Coordinates
				Eigen::Matrix<T, 2, 1> actualPixel = toNormalizedDeviceCoordinates(point.data());
				std::cout << actualPixel << std::endl;

				// TODO Normalized Device Coordinates -> Pixels

				std::cout << actualPixel << std::endl;
				return true;
			}
		};

		class CameraPoseEstimator {
		protected:
			// Build the problem.
			ceres::Problem problem;

			ceres::Solver::Options options;
			ceres::Solver::Summary summary;

			Eigen::Vector3d initialTranslation, translation;
			Eigen::Vector3d initialRotation, rotation;

		public:
			explicit CameraPoseEstimator(const Eigen::Vector3d &_initialTranslation,
										 const Eigen::Vector3d &_initialRotation);

			void addReprojectionResidual(const Eigen::Vector3d &worldCoordinate, const Eigen::Vector2d &pixel);

			void addReprojectionResidual(const Eigen::Vector4d &worldCoordinate, const Eigen::Vector2d &pixel);

			void solve();

		};
	}
}

#endif //CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP
