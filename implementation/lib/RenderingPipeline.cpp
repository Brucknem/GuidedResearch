//
// Created by brucknem on 04.02.21.
//

#include "RenderingPipeline.hpp"
#include "ceres/ceres.h"

namespace providentia {
	namespace camera {

		template<typename T>
		Eigen::Matrix<T, 2, 1>
		render(const T *_translation, const T *_rotation, const Eigen::Matrix<double, 3, 4> &_intrinsics, const T
		*vector) {
			bool flipped;
			return render(_translation, _rotation, _intrinsics, vector, flipped);
		}

		template<typename T>
		Eigen::Matrix<T, 2, 1>
		render(const T *_translation, const T *_rotation, const Eigen::Matrix<double, 3, 4> &_intrinsics, const T
		*vector, bool &flipped) {
			Eigen::Matrix<T, 3, 1> translation{_translation[0], _translation[1], _translation[2]};
			Eigen::Matrix<T, 3, 1> rotation{_rotation[0], _rotation[1], _rotation[2]};
			Eigen::Matrix<T, 3, 1> _vector{vector[0], vector[1], vector[2]};

			Eigen::Matrix<T, 4, 1> pointInCameraSpace = toCameraSpace(_translation, _rotation, vector);

			Eigen::Matrix<T, 3, 1> homogeneousPixel = _intrinsics.template cast<T>() * pointInCameraSpace;

			Eigen::Matrix<T, 2, 1> pixel = perspectiveDivision(homogeneousPixel, flipped);

			return pixel;
		}

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

			const Eigen::Matrix<T, 4, 4> &rotationMatrix = rotation * zAxis * yAxis * xAxis;
			return rotationMatrix;
		}

		template<typename T>
		Eigen::Matrix<T, 2, 1> perspectiveDivision(const Eigen::Matrix<T, 3, 1> &vector, bool &flipped) {
			flipped = vector[2] < (T) 0;
			Eigen::Matrix<T, 3, 1> result = vector / (vector[2] + 1e-51);
			return Eigen::Matrix<T, 2, 1>{result(0, 0), result(1, 0)};
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

		Eigen::Matrix<double, 2, 1> render(const Eigen::Vector3d &_translation, const Eigen::Vector3d &_rotation,
										   const Eigen::Matrix<double, 3, 4> &_intrinsics,
										   const Eigen::Vector4d &vector,
										   const cv::Vec3d &color, cv::Mat &image) {
			bool flipped;
			return render(_translation, _rotation, _intrinsics, vector, color, image, flipped);
		}

		Eigen::Matrix<double, 2, 1> render(const Eigen::Vector3d &_translation, const Eigen::Vector3d &_rotation,
										   const Eigen::Matrix<double, 3, 4> &_intrinsics,
										   const Eigen::Vector4d &vector,
										   const cv::Vec3d &color, cv::Mat &image, bool &flipped) {
			Eigen::Vector2i imageSize(image.cols, image.rows);

			Eigen::Vector2d pointInImageSpace = render(
				_translation.data(), _rotation.data(), _intrinsics,
				vector.data(), flipped);

			if (flipped) {
				return pointInImageSpace;
			}
			int imageHeight = imageSize.y() - 1;

			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < 2; ++j) {
					Eigen::Vector2i nearestPixel = pointInImageSpace.cast<int>();
					nearestPixel.x() += i;
					nearestPixel.y() += j;
					double distance = (nearestPixel.cast<double>() - pointInImageSpace).norm();
					nearestPixel.y() = imageHeight - nearestPixel.y();

//					std::cout << "[" << nearestPixel.x() << ", " << nearestPixel.y() << "] - " << distance;
					if (nearestPixel.x() >= imageSize.x() || nearestPixel.y() >= imageSize.y() ||
						nearestPixel.x() < 0 || nearestPixel.y() < 0) {
						continue;
					}
//					std::cout << std::endl;

					cv::Vec4d _color = {color[0], color[1], color[2], (distance / (double) sqrt(2))};

					// 1200, 1920
					image.at<cv::Vec4d>(nearestPixel.y(), nearestPixel.x()) = _color;
				}
			}

			cv::circle(image,
					   {(int) pointInImageSpace.x(), (int) (imageHeight - pointInImageSpace.y())},
					   std::min(5, std::max(0, (int) (imageSize.y() * 0.01))),
					   color,
					   cv::FILLED
			);
			return pointInImageSpace;
		}

		template<typename T>
		Eigen::Matrix<T, 3, 4> getIntrinsicsMatrix(const T *intrinsics) {
			T zero = (T) 0;
			Eigen::Matrix<T, 3, 4> intrinsicsMatrix = Eigen::Matrix<T, 3, 4>::Zero();

			T focalLength = intrinsics[0];

			Eigen::Matrix<T, 2, 1> m = Eigen::Matrix<T, 2, 1>(
				intrinsics[1],
				intrinsics[2]
			);

			Eigen::Matrix<T, 2, 1> principalPoint = Eigen::Matrix<T, 2, 1>(
				intrinsics[3],
				intrinsics[4]
			);

			T skew = intrinsics[5];

			Eigen::Matrix<T, 2, 1> alpha = focalLength * m.cwiseInverse();

			std::vector<T> values{
				alpha(0, 0), skew, principalPoint(0, 0),
				zero, alpha(1, 0), principalPoint(1, 0),
				zero, zero, (T) 1
			};

			return getIntrinsicsMatrixFromConfig(values.data());
		}

		template<typename T>
		Eigen::Matrix<T, 3, 4> getIntrinsicsMatrixFromConfig(const T *intrinsics) {
			Eigen::Matrix<T, 3, 4> matrix;
			T zero = T(0);
			matrix <<
				   intrinsics[0], intrinsics[1], intrinsics[2], zero,
				intrinsics[3], intrinsics[4], intrinsics[5], zero,
				intrinsics[6], intrinsics[7], intrinsics[8], zero;
//			std::cout << matrix << std::endl;
			return matrix;
		}

		Eigen::Matrix<double, 3, 4> getBlenderCameraIntrinsics() {
			double pixelWidth = 32. / 1920.;
			double principalX = 1920. / 2;
			double principalY = 1200. / 2;
			std::vector<double> _intrinsics{
				20, pixelWidth, pixelWidth, principalX, principalY, 0
			};
			return getIntrinsicsMatrix(_intrinsics.data());
		}

		Eigen::Matrix<double, 3, 4> getS40NCamFarIntrinsics() {
			return getIntrinsicsMatrixFromConfig<double>(new double[9]{
				9023.482825, 0.000000, 1222.314303, 0.000000, 9014.504360, 557.541182, 0.000000, 0.000000, 1.000000});
		}

#pragma region TemplateInstances

		//////////////////////////////////////////////////////////////////////////////////////////////////
		/// These template instances are needed to tell the compiler for which types we want specialized
		/// implementations of the functions.
		//////////////////////////////////////////////////////////////////////////////////////////////////

		template Eigen::Matrix<double, 2, 1>
		render<double>(const double *_translation, const double *_rotation,
					   const Eigen::Matrix<double, 3, 4> &_intrinsics,
					   const double *vector, bool &);

		template Eigen::Matrix<double, 2, 1>
		render<double>(const double *_translation, const double *_rotation,
					   const Eigen::Matrix<double, 3, 4> &_intrinsics,
					   const double *vector);

		/**
		 * ParametricPoint correspondences.
		 */
		template Eigen::Matrix<ceres::Jet<double, 9>, 2, 1>
		render<ceres::Jet<double, 9>>(const ceres::Jet<double, 9> *, const ceres::Jet<double, 9> *,
									  const Eigen::Matrix<double, 3, 4> &,
									  const ceres::Jet<double, 9> *, bool &);

#pragma endregion TemplateInstances

	}
}