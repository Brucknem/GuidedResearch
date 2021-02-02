//
// Created by brucknem on 02.02.21.
//

#include "CameraPoseEstimation.hpp"

#include <utility>
#include "Eigen/Dense"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ceres/autodiff_cost_function.h"

namespace providentia {
	namespace calibration {

		CameraPoseEstimator::CameraPoseEstimator(const camera::Camera &_camera) : camera(_camera) {};

		void CameraPoseEstimator::solve() {
			options.linear_solver_type = ceres::DENSE_QR;
			options.minimizer_progress_to_stdout = true;
			Solve(options, &problem, &summary);
		}

		void CameraPoseEstimator::addReprojectionResidual(const Eigen::Vector2i &pixel,
														  const Eigen::Vector3d &worldCoordinate) {
			ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<ReprojectionResidual, 3, 3, 2>(
					new ReprojectionResidual(pixel, worldCoordinate));
//			problem.AddResidualBlock(cost_function, nullptr, camera.getTranslation().data(),
//									 camera.getRotation().data());
		}

	}
}
