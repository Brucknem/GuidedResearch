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
			ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<ReprojectionResidual, 2, 3, 3>(
					new ReprojectionResidual(pixel, worldCoordinate, frustumParameters, intrinsics, imageSize));
			problem.AddResidualBlock(cost_function, nullptr, translation.data(), rotation.data());
		}

		CameraPoseEstimator::CameraPoseEstimator(Eigen::Vector3d _initialTranslation,
												 Eigen::Vector3d _initialRotation,
												 Eigen::Vector2d _frustumParameters,
												 Eigen::Vector3d _intrinsics,
												 Eigen::Vector2d _imageSize) :
//				initialRotation(std::move(_initialRotation)), initialTranslation(std::move(_initialTranslation)),
				frustumParameters(std::move(_frustumParameters)), intrinsics(std::move(_intrinsics)),
				imageSize(std::move(_imageSize)) {

			rotation.push_back(_initialRotation.x());
			rotation.push_back(_initialRotation.y());
			rotation.push_back(_initialRotation.z());

			translation.push_back(_initialTranslation.x());
			translation.push_back(_initialTranslation.y());
			translation.push_back(_initialTranslation.z());
		}

	}
}
