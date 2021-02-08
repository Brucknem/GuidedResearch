//
// Created by brucknem on 02.02.21.
//

#include "CameraPoseEstimation.hpp"

#include <utility>
#include "ceres/autodiff_cost_function.h"

namespace providentia {
	namespace calibration {

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

		void CameraPoseEstimator::addPointCorrespondence(const Eigen::Vector3d &worldPosition,
														 const Eigen::Vector2d &pixel) {
			double lambda = 0;
			double mu = 0;
			positions.emplace_back(worldPosition);
			problem.AddResidualBlock(
					CorrespondenceResidual::Create(pixel,
												   std::make_shared<providentia::calibration::Point>(positions.back()),
												   frustumParameters,
												   intrinsics,
												   imageSize),
					nullptr, translation.data(), rotation.data(), &lambda, &mu);
		}

		void CameraPoseEstimator::addLineCorrespondence(Eigen::Vector3d _origin, const Eigen::Vector3d &_heading,
														const Eigen::Vector2d &pixel) {

		}

		void CameraPoseEstimator::addPlaneCorrespondence(Eigen::Vector3d _origin, const Eigen::Vector3d &_axisA,
														 const Eigen::Vector3d &_axisB, const Eigen::Vector2d &pixel) {

		}

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
			for (const auto &worldPosition : positions) {
				meanVector += worldPosition.getPosition();
			}
			return meanVector / positions.size();
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

	}
}
