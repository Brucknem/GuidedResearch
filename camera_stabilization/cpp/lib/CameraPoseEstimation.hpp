//
// Created by brucknem on 02.02.21.
//

#ifndef CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP
#define CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP

#include <utility>

#include "Camera.hpp"
#include "ceres/ceres.h"

namespace providentia {
	namespace calibration {

		struct ReprojectionResidual {
		private:
			const Eigen::Vector2i pixel;
			const Eigen::Vector3d worldCoordinate;

		public:
			ReprojectionResidual(Eigen::Vector2i _pixel, Eigen::Vector3d _worldCoordinate) :
					pixel(std::move(_pixel)), worldCoordinate(std::move(_worldCoordinate)) {}

			template<typename T>
			bool operator()(const T *_translation, const T *_rotation, T *residual) const {
//				Eigen::Matrix<T, 4, 1> _point = {(T) worldCoordinate.x(), (T) worldCoordinate.y(),
//												 (T) worldCoordinate.z(), (T) 1};
//
//				Eigen::Matrix<T, 4, 1> rotationMatrix = providentia::camera::getRotationMatrix<T>(
//						{_rotation[0], _rotation[1], _rotation[2]});
//
//				Eigen::Matrix<T, 4, 1> translatedPoint =
//						_point - Eigen::Matrix<T, 4, 1>(_translation[0], _translation[1], _translation[2], 0);
//				_point = rotationMatrix.inverse() * translatedPoint;
//
//				Eigen::Vector3d pointInCameraSpace{_point(0, 0), _point(1, 0), _point(2, 0)};

				return true;
			}
		};

		class CameraPoseEstimator {
		protected:
			providentia::camera::Camera camera;

			// Build the problem.
			ceres::Problem problem;

			ceres::Solver::Options options;
			ceres::Solver::Summary summary;

		public:
			explicit CameraPoseEstimator(const providentia::camera::Camera &_camera);

			void addReprojectionResidual(const Eigen::Vector2i &pixel, const Eigen::Vector3d &worldCoordinate);

			void solve();
		};
	}
}

#endif //CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP
