//
// Created by brucknem on 02.02.21.
//

#include "CameraPoseEstimation.hpp"

#include <utility>
#include "ceres/autodiff_cost_function.h"
#include <limits>

namespace providentia {
	namespace calibration {

		CameraPoseEstimator::CameraPoseEstimator(Eigen::Matrix<double, 3, 4> intrinsics, bool initLogging,
												 double weightPenalizeScale) :
			intrinsics(std::move(intrinsics)), weightPenalizeScale(weightPenalizeScale) {
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

		Eigen::Vector3d CameraPoseEstimator::calculateFurthestPoint(const Eigen::Vector3d &mean) {
			Eigen::Vector3d furthestPoint{0, 0, 0};
			double maxDistance = 0;
			for (const auto &worldObject : worldObjects) {
				for (const auto &point : worldObject.getPoints()) {
					double distance = (point.getPosition() - mean).norm();
					if (distance > maxDistance) {
						maxDistance = distance;
						furthestPoint = point.getPosition();
					}
				}
			}
			return furthestPoint;
		}

		std::thread CameraPoseEstimator::estimateAsync(bool logSummary) {
			auto thread = std::thread(&CameraPoseEstimator::estimate, this, logSummary);
			thread.detach();
			return thread;
		}

		void CameraPoseEstimator::solveProblem(bool logSummary) {
			auto problem = createProblem();
			auto options = setupOptions(logSummary);
			Solve(options, &problem, &summary);
			evaluateAllResiduals(problem);
			evaluateCorrespondenceResiduals(problem);
			evaluateLambdaResiduals(problem);
			evaluateRotationResiduals(problem);
			evaluateWeightResiduals(problem);
		}

		void CameraPoseEstimator::estimate(bool logSummary) {
			optimizationFinished = false;
			foundValidSolution = false;
			int i = 0;
			for (; i < maxTriesUntilAbort; i++) {
				calculateInitialGuess();
				solveProblem(logSummary);
				double originalPenalize = lambdaPenalizeScale;
				lambdaPenalizeScale = originalPenalize * 10;
				solveProblem(logSummary);
				if (lambdasLoss > 10) {
					// > 10 is an empirical number. Might be further investigated.
					lambdaPenalizeScale = originalPenalize;
					continue;
				}
				correspondencesLoss = log10(correspondencesLoss);
				if (correspondencesLoss <= 1 || correspondencesLoss > weightsScale) {
					lambdaPenalizeScale = originalPenalize;
					continue;
				}
				lambdaPenalizeScale = originalPenalize * 100000;
				solveProblem(logSummary);
				lambdaPenalizeScale = originalPenalize;
				break;
			}
			foundValidSolution = i < maxTriesUntilAbort;
			optimizationFinished = true;
			if (logSummary) {
				std::cout << *this << std::endl;
			}

		}

		ceres::Solver::Options CameraPoseEstimator::setupOptions(bool logSummary) {
			ceres::Solver::Options options;
			options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//			options.trust_region_strategy_type = ceres::DOGLEG;
//			options.use_nonmonotonic_steps = true;
//			options.max_num_consecutive_invalid_steps = 15;
//			options.max_num_iterations = weights.size();
			options.max_num_iterations = 1000;
			options.num_threads = 12;
			options.minimizer_progress_to_stdout = logSummary;
			options.update_state_every_iteration = true;
			return options;
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

		ceres::Problem CameraPoseEstimator::createProblem() {
			auto problem = ceres::Problem();
			for (auto p : weights) {
				delete p;
			}
			weights.clear();
			correspondenceResiduals.clear();
			weightResiduals.clear();
			lambdaResiduals.clear();

			for (const auto &worldObject : worldObjects) {
				for (const auto &point : worldObject.getPoints()) {
					if (!point.hasExpectedPixel()) {
						continue;
					}
					*point.getLambda() = 0;
					*point.getMu() = 0;
					weights.emplace_back(new double(1));
					correspondenceResiduals.emplace_back(problem.AddResidualBlock(
						CorrespondenceResidual::create(
							point.getExpectedPixel(),
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
						point.getLambda(),
						point.getMu(),
						weights[weights.size() - 1]
					));

					lambdaResiduals.emplace_back(problem.AddResidualBlock(
						DistanceFromIntervalResidual::create(worldObject.getHeight()),
						getScaledHuberLoss(lambdaPenalizeScale),
						point.getLambda()
					));

					weightResiduals.emplace_back(problem.AddResidualBlock(
						DistanceResidual::create(1),
						getScaledHuberLoss(weightPenalizeScale),
						weights[weights.size() - 1]
					));
				}
			}
			// +1.5 empirical knowledge. Might be further investigated.
			// Better might be max(width) of all objects, i.e. max # of pixels per row over all objects.
			// weightScale = max(width) * weight.size()
			weightsScale = log10(weights.size() * 2) + 1.5;

//			addTranslationConstraints();
			addRotationConstraints(problem);

			std::cout << "Residuals: " << problem.NumResidualBlocks() << std::endl;
			return problem;
		}

		void CameraPoseEstimator::addRotationConstraints(ceres::Problem &problem) {
			rotationResiduals.clear();
			rotationResiduals.emplace_back(problem.AddResidualBlock(
				DistanceFromIntervalResidual::create(60, 110),
				getScaledHuberLoss(rotationPenalizeScale),
				&rotation.x()
			));
			rotationResiduals.emplace_back(problem.AddResidualBlock(
				DistanceFromIntervalResidual::create(-10, 10),
				getScaledHuberLoss(rotationPenalizeScale),
				&rotation.y()
			));
		}

		void CameraPoseEstimator::addTranslationConstraints(ceres::Problem &problem) {
//			int x_interval = 5000;
//			int y_interval = 5000;
//			int z_interval = 300;
			int scale = 10;
//			problem.AddResidualBlock(
//				DistanceFromIntervalResidual::create(translation.x() - x_interval, translation.x() + x_interval),
//				getScaledHuberLoss(scale),
//				&translation.x()
//			);
//			problem.AddResidualBlock(
//				DistanceFromIntervalResidual::create(translation.y() - y_interval, translation.y() + y_interval),
//				getScaledHuberLoss(scale),
//				&translation.y()
//			);
			problem.AddResidualBlock(
				DistanceFromIntervalResidual::create(0,
													 translation.z() + 1000),
				getScaledHuberLoss(scale),
				&translation.z()
			);
		}

		void CameraPoseEstimator::guessRotation(const Eigen::Vector3d &value) {
			hasRotationGuess = true;
			initialRotation = value;
			rotation = value;
		}

		void CameraPoseEstimator::guessTranslation(const Eigen::Vector3d &value) {
			hasTranslationGuess = true;
			initialTranslation = value;
			translation = value;
			initialDistanceFromMean = 0;
		}

		void CameraPoseEstimator::addWorldObject(const WorldObject &worldObject) {
			worldObjects.emplace_back(worldObject);
		}

		void CameraPoseEstimator::addWorldObjects(const std::vector<WorldObject> &vector) {
			for (const auto &worldObject : vector) {
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

		double CameraPoseEstimator::getWeightPenalizeScale() const {
			return weightPenalizeScale;
		}

		void CameraPoseEstimator::setWeightPenalizeScale(double value) {
			weightPenalizeScale = value;
		}

		void CameraPoseEstimator::clearWorldObjects() {
			worldObjects.clear();
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
			for (auto &worldObject : worldObjects) {
				for (auto &point : worldObject.getPoints()) {
					if (!point.hasExpectedPixel()) {
						continue;
					}
					result.emplace_back(*(point.getLambda()));
				}
			}
			return result;
		}

		double
		CameraPoseEstimator::evaluate(ceres::Problem &problem, const ceres::Problem::EvaluateOptions &evaluateOptions) {
			double loss;
			std::vector<double> residuals;
			problem.Evaluate(evaluateOptions, &loss, &residuals, nullptr, nullptr);
			return loss;
		}

		double CameraPoseEstimator::evaluate(ceres::Problem &problem, const std::vector<ceres::ResidualBlockId>
		&blockIds) {
			auto evaluateOptions = ceres::Problem::EvaluateOptions();
			evaluateOptions.residual_blocks = blockIds;
			return evaluate(problem, evaluateOptions);
		}

		void CameraPoseEstimator::evaluateAllResiduals(ceres::Problem &problem) {
			totalLoss = evaluate(problem);
		}

		void CameraPoseEstimator::evaluateCorrespondenceResiduals(ceres::Problem &problem) {
			correspondencesLoss = evaluate(problem, correspondenceResiduals);
		}

		void CameraPoseEstimator::evaluateLambdaResiduals(ceres::Problem &problem) {
			lambdasLoss = evaluate(problem, lambdaResiduals);
		}

		void CameraPoseEstimator::evaluateWeightResiduals(ceres::Problem &problem) {
			weightsLoss = evaluate(problem, weightResiduals);
		}

		void CameraPoseEstimator::evaluateRotationResiduals(ceres::Problem &problem) {
			rotationsLoss = evaluate(problem, rotationResiduals);
		}

		double CameraPoseEstimator::getLambdaPenalizeScale() const {
			return lambdaPenalizeScale;
		}

		void CameraPoseEstimator::setLambdaPenalizeScale(double value) {
			CameraPoseEstimator::lambdaPenalizeScale = value;
		}

		double CameraPoseEstimator::getRotationPenalizeScale() const {
			return rotationPenalizeScale;
		}

		void CameraPoseEstimator::setRotationPenalizeScale(double value) {
			rotationPenalizeScale = value;
		}

		bool CameraPoseEstimator::hasFoundValidSolution() const {
			return foundValidSolution;
		}

		double CameraPoseEstimator::getLambdasLoss() const {
			return lambdasLoss;
		}

		double CameraPoseEstimator::getCorrespondencesLoss() const {
			return correspondencesLoss;
		}

		double CameraPoseEstimator::getRotationsLoss() const {
			return rotationsLoss;
		}

		double CameraPoseEstimator::getWeightsLoss() const {
			return weightsLoss;
		}

		double CameraPoseEstimator::getTotalLoss() const {
			return totalLoss;
		}

		std::string printVectorRow(Eigen::Vector3d vector) {
			std::stringstream ss;
			ss << "[" << vector.x() << ", " << vector.y() << ", " << vector.z() << "]";
			return ss.str();
		}
	}
}
