//
// Created by brucknem on 02.02.21.
//

#ifndef CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP
#define CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP

#include <utility>

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"
#include "RenderingPipeline.hpp"

namespace providentia {
	namespace calibration {


		struct ReprojectionResidual {
		private:
			Eigen::Vector2d pixel;
			Eigen::Vector3d worldCoordinate;

			Eigen::Vector2d frustumParameters;
			Eigen::Vector3d intrinsics;

			Eigen::Vector2d imageSize;

		public:
			ReprojectionResidual(Eigen::Vector2d _pixel, Eigen::Vector3d _worldCoordinate,
								 Eigen::Vector2d _frustumParameters, Eigen::Vector3d _intrinsics,
								 Eigen::Vector2d _imageSize) :
					pixel(std::move(_pixel)), worldCoordinate(std::move(_worldCoordinate)),
					frustumParameters(std::move(_frustumParameters)), intrinsics(std::move(_intrinsics)),
					imageSize(std::move(_imageSize)) {}

			template<typename T>
			bool operator()(const T *_translation, const T *_rotation, T *residual) const {
				Eigen::Matrix<T, 4, 1> point{(T) worldCoordinate.x(), (T) worldCoordinate.y(), (T) worldCoordinate.z(),
											 (T) 1};

				Eigen::Matrix<T, 2, 1> _frustumParameters{(T) frustumParameters.x(), (T) frustumParameters.y()};
				Eigen::Matrix<T, 3, 1> _intrinsics{(T) intrinsics.x(), (T) intrinsics.y(), (T) intrinsics.z()};
				Eigen::Matrix<T, 2, 1> _imageSize{(T) imageSize.x(), (T) imageSize.y()};
				Eigen::Matrix<T, 2, 1> actualPixel;


//				point = providentia::camera::toCameraSpace(_translation, _rotation, point.data());
//				point = providentia::camera::toClipSpace(_frustumParameters.data(), _intrinsics.data(), point.data());
//				 actualPixel = providentia::camera::toNormalizedDeviceCoordinates(point.data());
//				actualPixel = providentia::camera::toImageSpace(_imageSize.data(), actualPixel.data());

				actualPixel = providentia::camera::render(_translation, _rotation, _frustumParameters.data(),
														  _intrinsics.data(), _imageSize.data(), point.data());

//				Eigen::Matrix<T, 2, 1> actualPixel{_translation[0], _rotation[1]};
				residual[0] = pixel.x() - actualPixel.x();
				residual[1] = pixel.y() - actualPixel.y();

				std::cout << residual[0] << ", " << residual[1] << std::endl;

				return true;
			}
		};

		class CameraPoseEstimator {
		protected:
			// Build the problem.
			ceres::Problem problem;

			ceres::Solver::Options options;
			ceres::Solver::Summary summary;

			std::vector<double> initialTranslation, translation;
			std::vector<double> initialRotation, rotation;

			Eigen::Vector2d frustumParameters;
			Eigen::Vector3d intrinsics;

			Eigen::Vector2d imageSize;

		public:
			explicit CameraPoseEstimator(Eigen::Vector3d _initialTranslation,
										 Eigen::Vector3d _initialRotation,
										 Eigen::Vector2d _frustumParameters,
										 Eigen::Vector3d _intrinsics,
										 Eigen::Vector2d _imageSize);

			void addReprojectionResidual(const Eigen::Vector3d &worldCoordinate, const Eigen::Vector2d &pixel);

			void addReprojectionResidual(const Eigen::Vector4d &worldCoordinate, const Eigen::Vector2d &pixel);

			void solve();

		};
	}
}

#endif //CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP
