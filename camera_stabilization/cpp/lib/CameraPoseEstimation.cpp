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

		void CameraPoseEstimator::calculateInitialGuess() {
			if (hasInitialGuessSet) {
				return;
			}
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
			for (const auto &worldObject : worldObjects) {
				for (const auto &point : worldObject.getPoints()) {
					meanVector += point->getPosition() * worldObject.getWeight();
				}
			}
			return meanVector / worldObjects.size();
		}

		void CameraPoseEstimator::estimate(bool _logSummary) {
			options.linear_solver_type = ceres::DENSE_QR;
			options.minimizer_progress_to_stdout = _logSummary;
			options.update_state_every_iteration = true;
			calculateInitialGuess();
			createProblem();
			Solve(options, &problem, &summary);
		}

		void CameraPoseEstimator::createProblem() {
			for (const auto &worldObject : worldObjects) {
				for (const auto &point : worldObject.getPoints()) {
					problem.AddResidualBlock(
							CorrespondenceResidual::Create(
									point->getExpectedPixel(),
									point,
									frustumParameters,
									intrinsics,
									imageSize,
									worldObject.getWeight()
							),
							nullptr,
							translation.data(),
							rotation.data(),
							point->getLambda(),
							point->getMu());
				}
			}
		}

		void CameraPoseEstimator::addIterationCallback(ceres::IterationCallback *callback) {
			options.callbacks.push_back(callback);
		}

		void CameraPoseEstimator::setInitialGuess(Eigen::Vector3d _translation, Eigen::Vector3d _rotation) {
			hasInitialGuessSet = true;
			initialTranslation = std::move(_translation);
			initialRotation = std::move(_rotation);
			translation = std::move(initialTranslation);
			rotation = std::move(initialRotation);
		}

		void CameraPoseEstimator::addWorldObject(const WorldObject &worldObject) {
			worldObjects.emplace_back(worldObject);
		}

		const std::vector<providentia::calibration::WorldObject> &CameraPoseEstimator::getWorldObjects() const {
			return worldObjects;
		}
	}
}
