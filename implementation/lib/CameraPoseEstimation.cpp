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

				initialTranslation = mean;
				initialTranslation.z() += initialDistanceFromMean;
				translation = initialTranslation;
			}

			if (!hasRotationGuess) {
				double x = 10.;
				initialRotation = {rng.uniform(-x, x), rng.uniform(-x, x), rng.uniform(-x, x)};
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
			calculateInitialGuess();
			createProblem();
			setupOptions(_logSummary);
			for (int i = 0; i < 5; ++i) {
				Solve(options, &problem, &summary);
				double finalLoss;
				problem.Evaluate(ceres::Problem::EvaluateOptions(), &finalLoss, nullptr, nullptr, nullptr);
				// TODO verify
				if (finalLoss >= weights.size() * 0.5 && finalLoss < 10. * weights.size()) {
					break;
				}
			}
			optimizationFinished = true;
			if (_logSummary) {
				std::cout << *this << std::endl;
			}
		}

		void CameraPoseEstimator::setupOptions(bool _logSummary) {
			options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//			options.trust_region_strategy_type = ceres::DOGLEG;
//			options.use_nonmonotonic_steps = true;
//			options.max_num_consecutive_invalid_steps = 15;
//			options.max_num_iterations = weights.size();
			options.max_num_iterations = 1000;
			options.num_threads = 12;
			options.minimizer_progress_to_stdout = _logSummary;
			options.update_state_every_iteration = true;
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
			correspondenceResiduals.clear();
			weightResiduals.clear();
			lambdaResiduals.clear();
			for (const auto &worldObject : worldObjects) {
				for (const auto &point : worldObject.getPoints()) {
					if (!point->hasExpectedPixel()) {
						continue;
					}
					weights.emplace_back(new double(1));
					correspondenceResiduals.emplace_back(problem.AddResidualBlock(
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
					));

					lambdaResiduals.emplace_back(problem.AddResidualBlock(
						DistanceFromIntervalResidual::Create(worldObject.getHeight()),
						getScaledHuberLoss(lambdaScale),
						point->getLambda()
					));

					weightResiduals.emplace_back(problem.AddResidualBlock(
						DistanceResidual::Create(1),
						getScaledHuberLoss(weightScale),
						weights[weights.size() - 1]
					));
				}
			}

//			addTranslationConstraints();
			addRotationConstraints();

			std::cout << "Residuals: " << problem.NumResidualBlocks() << std::endl;
		}

		void CameraPoseEstimator::addRotationConstraints() {
			rotationResiduals.clear();
			rotationResiduals.emplace_back(problem.AddResidualBlock(
				DistanceFromIntervalResidual::Create(60, 110),
				getScaledHuberLoss(rotationScale),
				&rotation.x()
			));
			rotationResiduals.emplace_back(problem.AddResidualBlock(
				DistanceFromIntervalResidual::Create(-10, 10),
				getScaledHuberLoss(rotationScale),
				&rotation.y()
			));
		}

		void CameraPoseEstimator::addTranslationConstraints() {
//			int x_interval = 5000;
//			int y_interval = 5000;
//			int z_interval = 300;
			int scale = 10;
//			problem.AddResidualBlock(
//				DistanceFromIntervalResidual::Create(translation.x() - x_interval, translation.x() + x_interval),
//				getScaledHuberLoss(scale),
//				&translation.x()
//			);
//			problem.AddResidualBlock(
//				DistanceFromIntervalResidual::Create(translation.y() - y_interval, translation.y() + y_interval),
//				getScaledHuberLoss(scale),
//				&translation.y()
//			);
			problem.AddResidualBlock(
				DistanceFromIntervalResidual::Create(0,
													 translation.z() + 1000),
				getScaledHuberLoss(scale),
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
			initialDistanceFromMean = 0;
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

		std::vector<double> CameraPoseEstimator::getLambdas() {
			std::vector<double> result;
			for (const auto &worldObject : worldObjects) {
				for (const auto &point : worldObject.getPoints()) {
					if (!point->hasExpectedPixel()) {
						continue;
					}
					result.emplace_back(*point->getLambda());
				}
			}
			return result;
		}

		double CameraPoseEstimator::evaluate(ceres::Problem::EvaluateOptions evaluateOptions) {
			double loss;
			std::vector<double> residuals;
			problem.Evaluate(evaluateOptions, &loss, &residuals, nullptr, nullptr);
			return loss;
		}

		double CameraPoseEstimator::evaluate(const std::vector<ceres::ResidualBlockId> &blockIds) {
			auto evaluateOptions = ceres::Problem::EvaluateOptions();
			evaluateOptions.residual_blocks = blockIds;
			return evaluate(evaluateOptions);
		}

		double CameraPoseEstimator::evaluateCorrespondenceResiduals() {
			return evaluate(correspondenceResiduals);
		}

		double CameraPoseEstimator::evaluateLambdaResiduals() {
			return evaluate(lambdaResiduals);
		}

		double CameraPoseEstimator::evaluateWeightResiduals() {
			return evaluate(weightResiduals);
		}

		double CameraPoseEstimator::evaluateRotationResiduals() {
			return evaluate(rotationResiduals);
		}

		double CameraPoseEstimator::getLambdaScale() const {
			return lambdaScale;
		}

		void CameraPoseEstimator::setLambdaScale(double lambdaScale) {
			CameraPoseEstimator::lambdaScale = lambdaScale;
		}

		double CameraPoseEstimator::getRotationScale() const {
			return rotationScale;
		}

		void CameraPoseEstimator::setRotationScale(double rotationScale) {
			CameraPoseEstimator::rotationScale = rotationScale;
		}

		std::string printVectorRow(Eigen::Vector3d vector) {
			std::stringstream ss;
			ss << "[" << vector.x() << ", " << vector.y() << ", " << vector.z() << "]";
			return ss.str();
		}
	}
}
