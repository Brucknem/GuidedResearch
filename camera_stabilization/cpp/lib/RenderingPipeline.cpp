//
// Created by brucknem on 04.02.21.
//

#include "RenderingPipeline.hpp"
#include "ceres/ceres.h"

namespace providentia {
	namespace camera {

		template<typename T>
		Eigen::Matrix<T, 4, 4> getCameraRotationMatrix(const T *_rotation) {
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
		Eigen::Matrix<T, 2, 1>
		render(const T *_translation, const T *_rotation, const T *_frustumParameters, const T *_intrinsics,
			   const T *_imageSize, const T *vector) {
			Eigen::Matrix<T, 4, 1> point = toCameraSpace(_translation, _rotation, vector);
//			if (point.z() < _frustumParameters[0]) {
//				return Eigen::Matrix<T, 2, 1>(_imageSize[0] * (T) 5, _imageSize[1] * (T) 5);
//			}
			point = toClipSpace(_frustumParameters, _intrinsics, point.data());

			Eigen::Matrix<T, 2, 1> pixel = toNormalizedDeviceCoordinates(point.data());
			pixel = toImageSpace(_imageSize, pixel.data());
			return pixel;
		}

		template<typename T>
		Eigen::Matrix<T, 2, 1> toImageSpace(const T *_imageSize, const T *vector) {
			Eigen::Matrix<T, 2, 1> pixel{vector[0], vector[1]};
			pixel += Eigen::Matrix<T, 2, 1>(1, 1);
			pixel *= (T) 0.5;
			return Eigen::Matrix<T, 2, 1>(pixel[0] * _imageSize[0], pixel[1] * _imageSize[1]);
		}

		template<typename T>
		Eigen::Matrix<T, 4, 1> toClipSpace(const T *_frustumParameters, const T *_intrinsics, const T *vector) {
			Eigen::Matrix<T, 4, 1> pointInCameraSpace{vector[0], vector[1], vector[2],
													  vector[3]};
			Eigen::Matrix<T, 4, 1> pointInClipSpace =
					getClipSpace(_frustumParameters, _intrinsics) * pointInCameraSpace;
			if (abs(pointInClipSpace.w()) <= (T) 1e-5) {
				return Eigen::Matrix<T, 4, 1>{(T) 0, (T) 0, (T) 1, (T) 1};
			}
			return normalize(pointInClipSpace);
		}

		template<typename T>
		Eigen::Matrix<T, 4, 4> getClipSpace(const T *_frustumParameters, const T *_intrinsics) {
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
			Eigen::Matrix<T, 4, 4> clipSpaceMatrix = normalization * getFrustum(_frustumParameters);
			return clipSpaceMatrix;
		}

		template<typename T>
		Eigen::Matrix<T, 4, 1> toFrustum(const T *_frustumParameters, const T *vector) {
			Eigen::Matrix<T, 4, 1> pointInFrustum = getFrustum(_frustumParameters) *
													Eigen::Matrix<T, 4, 1>{vector[0], vector[1], vector[2],
																		   vector[3]};
			return normalize(pointInFrustum);
		}

		template<typename T>
		Eigen::Matrix<T, 4, 1> normalize(const Eigen::Matrix<T, 4, 1> &vector) {
			return vector / (vector[3] + 1e-8);
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
		Eigen::Matrix<T, 4, 1> toCameraSpace(const T *_translation, const T *_rotation, const T *vector) {
			// TODO Remove inverse
			return getCameraRotationMatrix<T>(_rotation).inverse() *
				   (Eigen::Matrix<T, 4, 1>(
						   vector[0] - _translation[0],
						   vector[1] - _translation[1],
						   vector[2] - _translation[2],
						   (T) 1));
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

		template<typename T>
		Eigen::Matrix<T, 2, 1> toNormalizedDeviceCoordinates(const T *vector) {
			return Eigen::Matrix<T, 2, 1>(vector[0], vector[1]);
		}

#pragma region TemplateInstances

		//////////////////////////////////////////////////////////////////////////////////////////////////
		/// These template instances are needed to tell the compiler for which types we want specialized
		/// implementations of the functions.
		//////////////////////////////////////////////////////////////////////////////////////////////////

		template Eigen::Matrix<double, 2, 1> render<double>(const double *, const double *, const double *,
															const double *, const double *, const double *);

		template Eigen::Matrix<double, 4, 1> toFrustum(const double *, const double *);

		/**
		 * Point correspondences.
		 */
		template Eigen::Matrix<ceres::Jet<double, 6>, 2, 1>
		render<ceres::Jet<double, 6>>(const ceres::Jet<double, 6> *, const ceres::Jet<double, 6> *,
									  const ceres::Jet<double, 6> *,
									  const ceres::Jet<double, 6> *, const ceres::Jet<double, 6> *,
									  const ceres::Jet<double, 6> *);

		/**
		 * PointOnLine correspondences.
		 */
		template Eigen::Matrix<ceres::Jet<double, 7>, 2, 1>
		render<ceres::Jet<double, 7>>(const ceres::Jet<double, 7> *, const ceres::Jet<double, 7> *,
									  const ceres::Jet<double, 7> *,
									  const ceres::Jet<double, 7> *, const ceres::Jet<double, 7> *,
									  const ceres::Jet<double, 7> *);

		/**
		 * PointOnPlane correspondences.
		 */
		template Eigen::Matrix<ceres::Jet<double, 8>, 2, 1>
		render<ceres::Jet<double, 8>>(const ceres::Jet<double, 8> *, const ceres::Jet<double, 8> *,
									  const ceres::Jet<double, 8> *,
									  const ceres::Jet<double, 8> *, const ceres::Jet<double, 8> *,
									  const ceres::Jet<double, 8> *);

#pragma endregion TemplateInstances

	}
}