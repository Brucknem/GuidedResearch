//
// Created by brucknem on 02.02.21.
//

#include "CameraPoseEstimation.hpp"

#include <utility>
#include "ceres/autodiff_cost_function.h"
#include <limits>
#include <thread>

namespace providentia {
	namespace calibration {

		CameraPoseEstimation::CameraPoseEstimation() = default;

		const Eigen::Vector3d &CameraPoseEstimation::getTranslation() const {
			return translation;
		}

		Eigen::Vector3d CameraPoseEstimation::clearRotation(const Eigen::Vector3d &rotation) {
			Eigen::Vector3d resultRotation = rotation;
			for (int i = 0; i < 3; i++) {
				while (resultRotation[i] < -180 || resultRotation[i] > 180) {
					if (resultRotation[i] < -180) {
						resultRotation[i] += 360;
					}
					if (resultRotation[i] > 180) {
						resultRotation[i] -= 360;
					}
				}
			}
			return resultRotation;
		}

		Eigen::Vector3d CameraPoseEstimation::getRotation() const {
			return clearRotation(rotation);
		}

		void CameraPoseEstimation::calculateInitialGuess() {
			if (!hasTranslationGuess) {
				Eigen::Vector3d mean = calculateMean();

				initialTranslation = mean;
				initialTranslation.z() += initialDistanceFromMean;
				translation = initialTranslation;
			}

			if (!hasRotationGuess) {
				double x = 35.;
				initialRotation = {rng.uniform(-x, x), rng.uniform(-x, x), rng.uniform(-x, x)};
				rotation = initialRotation;
			}
		}

		Eigen::Vector3d CameraPoseEstimation::calculateMean() {
			Eigen::Vector3d meanVector(0, 0, 0);
			for (const auto &worldObject : worldObjects) {
				meanVector += worldObject.getMean();
			}
			return meanVector / worldObjects.size();
		}

		std::thread CameraPoseEstimation::estimateAsync(bool logSummary) {
			auto thread = std::thread(&CameraPoseEstimation::estimate, this, logSummary);
			thread.detach();
			return thread;
		}

		void CameraPoseEstimation::solveProblem(bool logSummary) {
			auto problem = createProblem();
			auto options = setupOptions(logSummary);
			Solve(options, &problem, &summary);
			evaluateAllResiduals(problem);
			evaluateCorrespondenceResiduals(problem);
			evaluateLambdaResiduals(problem);
			evaluateRotationResiduals(problem);
			evaluateWeightResiduals(problem);
			evaluateIntrinsicsResiduals(problem);
		}

		std::vector<double> CameraPoseEstimation::getLambdas() {
			std::vector<double> lambdas;
			for (const auto &worldObject : worldObjects) {
				std::cout << "World Object: " << worldObject.getId() << std::endl;
				for (const auto &point : worldObject.getCenterLine()) {
					std::cout << *point.getLambda() << " / " << worldObject.getHeight() << std::endl;
					lambdas.emplace_back(*point.getLambda());
				}
			}
			return lambdas;
		}

		void CameraPoseEstimation::estimate(bool logSummary) {
			if (!intrinsicsGuessed) {
				std::cout << "No intrinsics guessed. Aborting!" << std::endl;
			}
			optimizationFinished = false;
			foundValidSolution = false;
			int i = 0;
			for (; i < maxTriesUntilAbort; i++) {
				calculateInitialGuess();
				solveProblem(logSummary);
				double originalPenalize = lambdaResidualScalingFactor;
				lambdaResidualScalingFactor = originalPenalize * 10;
				solveProblem(logSummary);
				if (lambdasLoss > 5 || rotationsLoss > 0.5) {
					// > 10 is an empirical number. Might be further investigated.
					lambdaResidualScalingFactor = originalPenalize;
					continue;
				}
//				log10(correspondencesLoss);
//				std::cout << correspondencesLossScale << std::endl;
				if (correspondencesLoss > correspondenceLossUpperBound || rotationsLoss > 1e-6) {
					lambdaResidualScalingFactor = originalPenalize;
					continue;
				}
//				lambdaResidualScalingFactor = originalPenalize * 100;
				solveProblem(logSummary);
				lambdaResidualScalingFactor = originalPenalize;
//				auto lambdas = getLambdas();
				break;
			}
			foundValidSolution = i < maxTriesUntilAbort;
			optimizationFinished = true;
			if (logSummary) {
				std::cout << *this << std::endl;
			}

		}

		ceres::Solver::Options CameraPoseEstimation::setupOptions(bool logSummary) {
			ceres::Solver::Options options;
			options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//			options.trust_region_strategy_type = ceres::DOGLEG;
//			options.use_nonmonotonic_steps = true;
//			options.max_num_consecutive_invalid_steps = 15;
//			options.max_num_iterations = weights.size();
			options.max_num_iterations = 1000;
//			options.num_threads = 1;
			auto processorCount = std::thread::hardware_concurrency();
			if (processorCount == 0) {
				processorCount = 8;
			}
			options.num_threads = processorCount;
			options.minimizer_progress_to_stdout = logSummary;
			options.update_state_every_iteration = true;
			return options;
		}

		ceres::ScaledLoss *CameraPoseEstimation::getScaledHuberLoss(double scale) {
			return getScaledHuberLoss(1.0, scale);
		}

		ceres::ScaledLoss *CameraPoseEstimation::getScaledHuberLoss(double huber, double scale) {
			return new ceres::ScaledLoss(
				new ceres::HuberLoss(huber),
				scale,
				ceres::TAKE_OWNERSHIP
			);
		}

		ceres::Problem CameraPoseEstimation::createProblem() {
			auto problem = ceres::Problem();
			for (auto p : weights) {
				delete p;
			}
			weights.clear();
			correspondenceResiduals.clear();
			weightResiduals.clear();
			lambdaResiduals.clear();
			intrinsicsResiduals.clear();

			for (auto &worldObject : worldObjects) {
				for (const auto &point : worldObject.getCenterLine()) {
					*point.getLambda() = 0;
					*point.getMu() = 0;
					weights.emplace_back(new double(1));
					correspondenceResiduals.emplace_back(addCorrespondenceResidualBlock(problem, point));
					lambdaResiduals.emplace_back(addLambdaResidualBlock(problem, point, worldObject.getHeight()));
					weightResiduals.emplace_back(addWeightResidualBlock(problem, weights[weights.size() - 1]));
				}
			}
			// +1.5 empirical knowledge. Might be further investigated.
			// Better might be max(width) of all objects, i.e. max # of pixels per row over all objects.
			// weightScale = max(width) * weight.size()
			correspondenceLossUpperBound = weightResiduals.size();

			addRotationConstraints(problem);
			addIntrinsicsConstraints(problem);

//			std::cout << "Residuals: " << problem.NumResidualBlocks() << std::endl;
			return problem;
		}

		ceres::ResidualBlockId
		CameraPoseEstimation::addWeightResidualBlock(ceres::Problem &problem, double *weight) const {
			return problem.AddResidualBlock(
				residuals::DistanceResidual::create(1),
				getScaledHuberLoss(weightResidualScalingFactor),
				weight
			);
		}

		ceres::ResidualBlockId
		CameraPoseEstimation::addLambdaResidualBlock(ceres::Problem &problem, const ParametricPoint &point,
													 double height) const {
			return problem.AddResidualBlock(
				residuals::DistanceFromIntervalResidual::create(height),
				getScaledHuberLoss(lambdaResidualScalingFactor),
				point.getLambda()
			);
		}

		ceres::ResidualBlockId
		CameraPoseEstimation::addCorrespondenceResidualBlock(ceres::Problem &problem, const ParametricPoint &point) {
			return problem.AddResidualBlock(
				residuals::CorrespondenceResidual::create(
					point.getExpectedPixel(),
					point
				),
				new ceres::HuberLoss(1.0),
				&intrinsics[0],
				&intrinsics[1],
				&intrinsics[2],
				&intrinsics[3],
				&intrinsics[4],
				&translation.x(),
				&translation.y(),
				&translation.z(),
				&rotation.x(),
				&rotation.y(),
				&rotation.z(),
				point.getLambda(),
				point.getMu(),
				weights[weights.size() - 1]
			);
		}

		void CameraPoseEstimation::addIntrinsicsConstraints(ceres::Problem &problem) {
			double scale = 10;
			double factor = 0.9;
			if (intrinsicsFixed) {
				factor = 1.;
			}

			intrinsicsResiduals.emplace_back(problem.AddResidualBlock(
				providentia::calibration::residuals::DistanceFromIntervalResidual::create(
					initialIntrinsics[0] * factor,
					initialIntrinsics[0] / factor),
				getScaledHuberLoss(scale),
				&intrinsics[0]
			));

			for (int i = 1; i < 5; ++i) {
				intrinsicsResiduals.emplace_back(problem.AddResidualBlock(
					providentia::calibration::residuals::DistanceResidual::create(
						initialIntrinsics[i]
					),
					getScaledHuberLoss(1e52),
					&intrinsics[i]
				));
			}
		}

		void CameraPoseEstimation::addRotationConstraints(ceres::Problem &problem) {
			rotationResiduals.clear();
			rotationResiduals.emplace_back(problem.AddResidualBlock(
				providentia::calibration::residuals::DistanceFromIntervalResidual::create(60, 110),
				getScaledHuberLoss(rotationResidualScalingFactor),
				&rotation.x()
			));
			rotationResiduals.emplace_back(problem.AddResidualBlock(
				providentia::calibration::residuals::DistanceFromIntervalResidual::create(-10, 10),
				getScaledHuberLoss(rotationResidualScalingFactor),
				&rotation.y()
			));
		}

		void CameraPoseEstimation::guessRotation(const Eigen::Vector3d &value) {
			hasRotationGuess = true;
			initialRotation = value;
			rotation = value;
		}

		void CameraPoseEstimation::guessTranslation(const Eigen::Vector3d &value) {
			hasTranslationGuess = true;
			initialTranslation = value;
			translation = value;
			initialDistanceFromMean = 0;
		}

		void CameraPoseEstimation::addWorldObject(const WorldObject &worldObject) {
			worldObjects.emplace_back(worldObject);
		}

		void CameraPoseEstimation::addWorldObjects(const std::vector<WorldObject> &vector) {
			for (const auto &worldObject : vector) {
				addWorldObject(worldObject);
			}
		}

		const std::vector<providentia::calibration::WorldObject> &CameraPoseEstimation::getWorldObjects() const {
			return worldObjects;
		}

		bool CameraPoseEstimation::isEstimationFinished() const {
			return optimizationFinished;
		}

		std::ostream &operator<<(std::ostream &os, const CameraPoseEstimation &estimator) {
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

		void CameraPoseEstimation::setWeightPenalizeScale(double value) {
			weightResidualScalingFactor = value;
		}

		void CameraPoseEstimation::clearWorldObjects() {
			worldObjects.clear();
		}

		std::vector<double> CameraPoseEstimation::getWeights() {
			std::vector<double> result;
			result.reserve(weights.size());
			std::transform(std::begin(weights), std::end(weights),
						   std::back_inserter(result), [](const double *weight) { return *weight; }
			);
			return result;
		}

		double
		CameraPoseEstimation::evaluate(ceres::Problem &problem,
									   const ceres::Problem::EvaluateOptions &evaluateOptions) {
			double loss;
			std::vector<double> residuals;
			problem.Evaluate(evaluateOptions, &loss, &residuals, nullptr, nullptr);
			return loss;
		}

		double CameraPoseEstimation::evaluate(ceres::Problem &problem, const std::vector<ceres::ResidualBlockId>
		&blockIds) {
			auto evaluateOptions = ceres::Problem::EvaluateOptions();
			evaluateOptions.residual_blocks = blockIds;
			return evaluate(problem, evaluateOptions);
		}

		void CameraPoseEstimation::evaluateAllResiduals(ceres::Problem &problem) {
			totalLoss = evaluate(problem);
		}

		void CameraPoseEstimation::evaluateCorrespondenceResiduals(ceres::Problem &problem) {
			correspondencesLoss = evaluate(problem, correspondenceResiduals);
		}

		void CameraPoseEstimation::evaluateLambdaResiduals(ceres::Problem &problem) {
			lambdasLoss = evaluate(problem, lambdaResiduals);
		}

		void CameraPoseEstimation::evaluateWeightResiduals(ceres::Problem &problem) {
			weightsLoss = evaluate(problem, weightResiduals);
		}

		void CameraPoseEstimation::evaluateRotationResiduals(ceres::Problem &problem) {
			rotationsLoss = evaluate(problem, rotationResiduals);
		}

		void CameraPoseEstimation::evaluateIntrinsicsResiduals(ceres::Problem &problem) {
			intrinsicsLoss = evaluate(problem, intrinsicsResiduals);
		}

		bool CameraPoseEstimation::hasFoundValidSolution() const {
			return foundValidSolution;
		}

		double CameraPoseEstimation::getLambdasLoss() const {
			return lambdasLoss;
		}

		double CameraPoseEstimation::getCorrespondencesLoss() const {
			return correspondencesLoss;
		}

		double CameraPoseEstimation::getRotationsLoss() const {
			return rotationsLoss;
		}

		double CameraPoseEstimation::getWeightsLoss() const {
			return weightsLoss;
		}

		double CameraPoseEstimation::getTotalLoss() const {
			return totalLoss;
		}

		const std::vector<double> &CameraPoseEstimation::getIntrinsics() const {
			return intrinsics;
		}

		void CameraPoseEstimation::guessIntrinsics(const std::vector<double> &values) {
			intrinsicsGuessed = true;
			intrinsics = values;
			initialIntrinsics = values;
		}

		void CameraPoseEstimation::fixIntrinsics(bool fixed) {
			intrinsicsFixed = fixed;
		}

		double CameraPoseEstimation::getIntrinsicsLoss() const {
			return intrinsicsLoss;
		}

		std::string printVectorRow(Eigen::Vector3d vector) {
			std::stringstream ss;
			ss << "[" << vector.x() << ", " << vector.y() << ", " << vector.z() << "]";
			return ss.str();
		}
	}
}
