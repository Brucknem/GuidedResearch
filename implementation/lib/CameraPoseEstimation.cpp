//
// Created by brucknem on 02.02.21.
//

#include "CameraPoseEstimation.hpp"

#include <utility>
#include "ceres/autodiff_cost_function.h"

namespace providentia {
	namespace calibration {

		CameraPoseEstimator::CameraPoseEstimator(Eigen::Matrix<double, 3, 4> _intrinsics, bool initLogging) :
			intrinsics(std::move(_intrinsics)) {
			if (initLogging) {
				google::InitGoogleLogging("Camera Pose Estimation");
			}
		}

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

		std::thread CameraPoseEstimator::estimateAsync(bool _logSummary) {
			auto thread = std::thread(&CameraPoseEstimator::estimate, this, _logSummary);
			thread.detach();
			return thread;
		}

		void CameraPoseEstimator::estimate(bool _logSummary) {
			optimizationFinished = false;
			options.linear_solver_type = ceres::DENSE_QR;
			options.minimizer_progress_to_stdout = _logSummary;
			options.update_state_every_iteration = true;
			calculateInitialGuess();
			createProblem();
			Solve(options, &problem, &summary);
			optimizationFinished = true;
			if (_logSummary) {
				std::cout << *this << std::endl;
			}
		}

		void CameraPoseEstimator::createProblem() {
			int i = 0;
			for (const auto &worldObject : worldObjects) {
				for (const auto &point : worldObject.getPoints()) {
					if (!point->hasExpectedPixel()) {
						continue;
					}
					i++;
					problem.AddResidualBlock(
						CorrespondenceResidual::Create(
							point->getExpectedPixel(),
							point,
							intrinsics,
							worldObject.getHeight(),
							worldObject.getWeight()
						),
						nullptr,
						translation.data(),
						rotation.data(),
						point->getLambda(),
						point->getMu());
				}
			}
			std::cout << "Added residuals: " << i << std::endl;
		}

		void CameraPoseEstimator::addIterationCallback(ceres::IterationCallback *callback) {
			options.callbacks.push_back(callback);
		}

		void
		CameraPoseEstimator::setInitialGuess(const Eigen::Vector3d &_translation, const Eigen::Vector3d &_rotation) {
			hasInitialGuessSet = true;
			initialTranslation = _translation;
			initialRotation = _rotation;
			translation = _translation;
			rotation = _rotation;
		}

		void CameraPoseEstimator::addWorldObject(const WorldObject &worldObject) {
			worldObjects.emplace_back(worldObject);
		}

		void CameraPoseEstimator::addWorldObjects(const std::vector<WorldObject> &_worldObjects) {
			for (const auto &worldObject : _worldObjects) {
				addWorldObject(worldObject);
			}
		}

		const std::vector<providentia::calibration::WorldObject> &CameraPoseEstimator::getWorldObjects() const {
			return worldObjects;
		}

		bool CameraPoseEstimator::isOptimizationFinished() const {
			return optimizationFinished;
		}
	}
}
