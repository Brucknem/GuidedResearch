//
// Created by brucknem on 02.02.21.
//

#include "CameraPoseEstimation.hpp"

#include <utility>
#include "ceres/autodiff_cost_function.h"

namespace providentia {
	namespace calibration {

		void CameraPoseEstimator::calculateInitialGuess() {
			Eigen::Vector3d mean = calculateMean();
			double wantedDistance = (0.5 * (frustumParameters.y() - frustumParameters.x())) + frustumParameters.x();

			initialTranslation = mean;
			initialTranslation.z() += wantedDistance;
			initialRotation = {0, 0, 0};

			translation = initialTranslation;
			rotation = initialRotation;
		}

		Eigen::Vector3d CameraPoseEstimator::calculateMean() {
			Eigen::Vector3d meanVector(0, 0, 0);
			for (const auto &worldPosition : worldPositions) {
				meanVector += worldPosition;
			}
			return meanVector / worldPositions.size();
		}

		void CameraPoseEstimator::estimate(bool _logSummary) {
			options.linear_solver_type = ceres::DENSE_QR;
			options.minimizer_progress_to_stdout = _logSummary;
			options.update_state_every_iteration = true;
			calculateInitialGuess();
			Solve(options, &problem, &summary);
		}

		void CameraPoseEstimator::addIterationCallback(ceres::IterationCallback *callback) {
			options.callbacks.push_back(callback);
		}

		void CameraPoseEstimator::addPointCorrespondence(const Eigen::Vector3d &worldPosition,
														 const Eigen::Vector2d &pixel) {
			worldPositions.push_back(worldPosition);
			problem.AddResidualBlock(
					PointCorrespondenceResidual::Create(pixel, worldPosition, frustumParameters, intrinsics, imageSize),
					nullptr, translation.data(), rotation.data()
			);
		}

		CameraPoseEstimator::CameraPoseEstimator(Eigen::Vector2d _frustumParameters,
												 Eigen::Vector3d _intrinsics,
												 Eigen::Vector2d _imageSize) :
				frustumParameters(std::move(_frustumParameters)), intrinsics(std::move(_intrinsics)),
				imageSize(std::move(_imageSize)) {}

		const Eigen::Vector3d &CameraPoseEstimator::getTranslation() const {
			return translation;
		}

		const Eigen::Vector3d &CameraPoseEstimator::getRotation() const {
			return rotation;
		}

	}
}
