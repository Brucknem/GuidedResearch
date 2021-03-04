//
// Created by brucknem on 02.02.21.
//

#include "CameraPoseEstimation.hpp"

#include <utility>
#include "ceres/autodiff_cost_function.h"

namespace providentia {
	namespace calibration {

		CameraPoseEstimator::CameraPoseEstimator(Eigen::Matrix<double, 3, 4> _intrinsics) :
			intrinsics(std::move(_intrinsics)) {}

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
//			Eigen::Vector3d mean = calculateMean();
//			Eigen::Vector3d furthestPoint = calculateFurthestPoint(mean);
//			Eigen::Vector3d lookAt = mean - furthestPoint;
//			initialTranslation = furthestPoint - lookAt.normalized() * (frustumParameters.x() + 1);
//
//			lookAt.z() = 0;
//			lookAt = lookAt.normalized();
//			double angle = Eigen::Vector3d(0, 1, 0).dot(lookAt);
//			angle = acos(angle) / M_PI * 180;
//
//			initialRotation = {90, 0, angle};
//
//			translation = initialTranslation;
//			rotation = initialRotation;
			Eigen::Vector3d mean = calculateMean();
			double wantedDistance = 500;

			initialTranslation = mean;
			initialTranslation.z() += wantedDistance;
			initialRotation = {0, 0, 0};

			translation = initialTranslation;
			rotation = initialRotation;
		}

		Eigen::Vector3d CameraPoseEstimator::calculateMean() {
			Eigen::Vector3d meanVector(0, 0, 0);
			for (const auto &worldObject : worldObjects) {
				meanVector += worldObject.getMean();
			}
			return meanVector / worldObjects.size();
		}

		Eigen::Vector3d CameraPoseEstimator::calculateFurthestPoint(Eigen::Vector3d mean) {
			Eigen::Vector3d furthestPoint{0, 0, 0};
			double maxDistance = 0;
			for (const auto &worldObject : worldObjects) {
				for (const auto &point : worldObject.getPoints()) {
					double distance = (point->getPosition() - mean).norm();
					if (distance > maxDistance) {
						maxDistance = distance;
						furthestPoint = point->getPosition();
					}
				}
			}
			return furthestPoint;
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
							intrinsics,
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
