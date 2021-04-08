//
// Created by brucknem on 02.02.21.
//

#include "CameraPoseEstimation.hpp"

#include <utility>
#include "ceres/autodiff_cost_function.h"
#include <limits>

namespace providentia {
	namespace calibration {

		CameraPoseEstimator::CameraPoseEstimator(Eigen::Matrix<double, 3, 4> _intrinsics, bool initLogging,
												 double weightScale) :
			intrinsics(std::move(_intrinsics)), weightScale(weightScale) {
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
			if (!hasTranslationGuess) {
				Eigen::Vector3d mean = calculateMean();
				double wantedDistance = 500;

				initialTranslation = mean;
				initialTranslation.z() += wantedDistance;
				translation = initialTranslation;
			}

			if (!hasRotationGuess) {
				initialRotation = {0, 0, 0};
				rotation = initialRotation;
			}
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
			for (int i = 0; i < 5; ++i) {
				Solve(options, &problem, &summary);
			}
			optimizationFinished = true;
			if (_logSummary) {
				std::cout << *this << std::endl;
			}
		}

		ceres::ScaledLoss *CameraPoseEstimator::getScaledHuberLoss(double scale) {
			return getScaledHuberLoss(1.0, scale);
		}

		ceres::ScaledLoss *CameraPoseEstimator::getScaledHuberLoss(double huber, double scale) {
			return new ceres::ScaledLoss(
				new ceres::HuberLoss(huber),
				scale,
				ceres::TAKE_OWNERSHIP
			);
		}

		void CameraPoseEstimator::createProblem() {
			problem = ceres::Problem();
			weights.clear();
			for (const auto &worldObject : worldObjects) {
//				weights.emplace_back(new double(1));
//				bool hasPoints = false;
				for (const auto &point : worldObject.getPoints()) {
					if (!point->hasExpectedPixel()) {
						continue;
					}
//					hasPoints = true;
					weights.emplace_back(new double(1));
					problem.AddResidualBlock(
						CorrespondenceResidual::Create(
							point->getExpectedPixel(),
							point,
							intrinsics
						),
						new ceres::HuberLoss(1.0),
						&translation.x(),
						&translation.y(),
						&translation.z(),
						&rotation.x(),
						&rotation.y(),
						&rotation.z(),
						point->getLambda(),
						point->getMu(),
						weights[weights.size() - 1]
					);

					problem.AddResidualBlock(
						DistanceFromIntervalResidual::Create(worldObject.getHeight()),
						getScaledHuberLoss(100),
						point->getLambda()
					);

					problem.AddResidualBlock(
						DistanceResidual::Create(1),
						getScaledHuberLoss(weightScale),
						weights[weights.size() - 1]
					);
				}

//				if (!hasPoints) {
//					weights.pop_back();
//					continue;
//				}
//
//				problem.AddResidualBlock(
//					DistanceResidual::Create(1),
//					new ceres::ScaledLoss(
//						nullptr,
////						std::numeric_limits<double>::max(),
//						1e5,
//						ceres::TAKE_OWNERSHIP
//					),
//					weights[weights.size() - 1]
//				);
			}

			addTranslationConstraints();
			addRotationConstraints();

			std::cout << "Residuals: " << problem.NumResidualBlocks() << std::endl;
		}

		void CameraPoseEstimator::addRotationConstraints() {
			problem.AddResidualBlock(
				DistanceFromIntervalResidual::Create(70, 100),
				getScaledHuberLoss(1000),
				&rotation.x()
			);
			problem.AddResidualBlock(
				DistanceFromIntervalResidual::Create(-10, 10),
				getScaledHuberLoss(1000),
				&rotation.y()
			);
		}

		void CameraPoseEstimator::addTranslationConstraints() {
			problem.AddResidualBlock(
				DistanceFromIntervalResidual::Create(translation.x() - 2000, translation.x() + 2000),
				getScaledHuberLoss(1000),
				&translation.x()
			);
			problem.AddResidualBlock(
				DistanceFromIntervalResidual::Create(translation.y() - 2000, translation.y() + 2000),
				getScaledHuberLoss(1000),
				&translation.y()
			);
			problem.AddResidualBlock(
				DistanceFromIntervalResidual::Create(translation.z() - 500 - 200, translation.z() - 500 + 200),
				getScaledHuberLoss(1000),
				&translation.z()
			);
		}

		void CameraPoseEstimator::addIterationCallback(ceres::IterationCallback *callback) {
			options.callbacks.push_back(callback);
		}

		void CameraPoseEstimator::guessRotation(const Eigen::Vector3d &_rotation) {
			hasRotationGuess = true;
			initialRotation = _rotation;
			rotation = _rotation;
		}

		void CameraPoseEstimator::guessTranslation(const Eigen::Vector3d &_translation) {
			hasTranslationGuess = true;
			initialTranslation = _translation;
			translation = _translation;
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

		std::ostream &operator<<(std::ostream &os, const CameraPoseEstimator &estimator) {
			os << "Translation:" << std::endl;
			os << "From:       " << printVectorRow(estimator.initialTranslation) << std::endl;
			os << "To:         " << printVectorRow(estimator.translation) << std::endl;
			os << "Difference: " << printVectorRow(estimator.translation - estimator.initialTranslation)
			   << std::endl;

			os << "Rotation:" << std::endl;
			os << "From:       " << printVectorRow(estimator.initialRotation) << std::endl;
			os << "To:         " << printVectorRow(estimator.rotation) << std::endl;
			os << "Difference: " << printVectorRow(estimator.rotation - estimator.initialRotation) << std::endl;

			os << "Weights:" << std::endl;
			for (const auto &weight : estimator.weights) {
				os << *weight << ", ";
			}
			return os;
		}

		double CameraPoseEstimator::getWeightScale() const {
			return weightScale;
		}

		void CameraPoseEstimator::setWeightScale(double weightScale) {
			CameraPoseEstimator::weightScale = weightScale;
		}

		void CameraPoseEstimator::clearWorldObjects() {
			worldObjects = std::vector<providentia::calibration::WorldObject>{};
		}

		std::vector<double> CameraPoseEstimator::getWeights() {
			std::vector<double> result;
			result.reserve(weights.size());
			std::transform(std::begin(weights), std::end(weights),
						   std::back_inserter(result), [](const double *weight) { return *weight; }
			);
			return result;
		}

		std::string printVectorRow(Eigen::Vector3d vector) {
			std::stringstream ss;
			ss << "[" << vector.x() << ", " << vector.y() << ", " << vector.z() << "]";
			return ss.str();
		}
	}
}
