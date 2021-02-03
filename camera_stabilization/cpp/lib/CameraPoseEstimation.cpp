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


		void CameraPoseEstimator::solve() {
			options.linear_solver_type = ceres::DENSE_QR;
			options.minimizer_progress_to_stdout = true;
			Solve(options, &problem, &summary);
		}

		void CameraPoseEstimator::addReprojectionResidual(const Eigen::Vector4d &worldCoordinate,
														  const Eigen::Vector2d &pixel) {
			addReprojectionResidual(Eigen::Vector3d(worldCoordinate.x(), worldCoordinate.y(), worldCoordinate.z()),
									pixel);
		}

		void CameraPoseEstimator::addReprojectionResidual(const Eigen::Vector3d &worldCoordinate,
														  const Eigen::Vector2d &pixel) {
			ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<ReprojectionResidual, 3, 3, 2>(
					new ReprojectionResidual(pixel, worldCoordinate));
			problem.AddResidualBlock(cost_function, nullptr, translation.data(), rotation.data());
		}

		CameraPoseEstimator::CameraPoseEstimator(const Eigen::Vector3d &_initialTranslation,
												 const Eigen::Vector3d &_initialRotation) {
			initialTranslation = _initialTranslation;
			initialRotation = _initialRotation;
			translation = _initialTranslation;
			rotation = _initialRotation;
		}

	}
}
