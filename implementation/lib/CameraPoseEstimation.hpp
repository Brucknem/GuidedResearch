//
// Created by brucknem on 02.02.21.
//

#ifndef CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP
#define CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP

#include <utility>
#include <vector>
#include <iostream>
#include <thread>
#include <limits>

#include "ceres/ceres.h"
#include "glog/logging.h"

#include "opencv2/opencv.hpp"
#include "Residuals.hpp"
#include "WorldObjects.hpp"

namespace providentia {
	namespace calibration {

		/**
		 * Prints a vector as a row.
		 */
		std::basic_string<char, std::char_traits<char>, std::allocator<char>> printVectorRow(Eigen::Vector3d vector);

		/**
		 * Estimates the camera pose from some known correspondences between the world and image.
		 */
		class CameraPoseEstimator {
		protected:

			/**
			 * The ceres internal minimization problem definition.
			 */
			ceres::Problem problem;

			/**
			 * Some options passed to the ceres solver.
			 */
			ceres::Solver::Options options;

			/**
			 * The final optimization summary.
			 */
			ceres::Solver::Summary summary;

			/**
			 * The initial camera [x, y, z] translation in world space.
			 */
			Eigen::Vector3d initialTranslation;

			/**
			 * The current camera [x, y, z] translation in world space used for optimization.
			 */
			Eigen::Vector3d translation;

			/**
			 * The initial camera [x, y, z] euler angle rotation around the world space axis.
			 */
			Eigen::Vector3d initialRotation;

			/**
			 * The current camera [x, y, z] euler angle rotation around the world space axis.
			 */
			Eigen::Vector3d rotation;

			/**
			 * The intrinsics matrix of the pinhole camera model.
			 */
			Eigen::Matrix<double, 3, 4> intrinsics;

			/**
			 * A buffer for the known world worldObjects.
			 */
			std::vector<providentia::calibration::WorldObject> worldObjects;

			bool hasInitialGuessSet = false;

			bool optimizationFinished = false;

			double weightScale;

			std::vector<double *> weights;

			std::vector<ceres::ResidualBlockId> residualBlockIds;

			std::vector<ceres::LossFunctionWrapper *> lossFunctions;

		public:
			/**
			 * @constructor
			 *
			 * @param _intrinsics The intrinsics of the pinhole camera model.
			 */
			CameraPoseEstimator(Eigen::Matrix<double, 3, 4> _intrinsics, bool initLogging = true,
								double weightScale = std::numeric_limits<double>::max());

			/**
			 * @destructor
			 */
			virtual ~CameraPoseEstimator() = default;

			void addWorldObject(const WorldObject &worldObject);

			void addWorldObjects(const std::vector<WorldObject> &worldObjects);

			/**
			 * Estimates the camera translation and rotation based on the known correspondences between the world and
			 * image.
			 */
			void estimate(bool _logSummary = false);

			std::thread estimateAsync(bool _logSummary = false);

			/**
			 * Based on the known world positions calculates and initial guess for the camera translation and rotation.
			 * This is necessary as the optimization problem is rather ill posed and sensitive to the initialization.
			 */
			void calculateInitialGuess();

			void setInitialGuess(const Eigen::Vector3d &translation, const Eigen::Vector3d &rotation);

			/**
			 * @get
			 */
			const Eigen::Vector3d &getTranslation() const;

			/**
			 * @get
			 */
			const Eigen::Vector3d &getRotation() const;

			const std::vector<providentia::calibration::WorldObject> &getWorldObjects() const;

			bool isOptimizationFinished() const;

			/**
			 * Calculates the mean of the known world correspondences.
			 */
			Eigen::Vector3d calculateMean();

			/**
			 * Adds a callback to the estimator that gets called after each optimization step.
			 */
			void addIterationCallback(ceres::IterationCallback *callback);

			void createProblem();

			Eigen::Vector3d calculateFurthestPoint(Eigen::Vector3d mean);

			friend std::ostream &operator<<(std::ostream &os, const CameraPoseEstimator &estimator);

			double getWeightScale() const;

			void setWeightScale(double weightScale);

			void clearWorldObjects() {
				worldObjects = std::vector<providentia::calibration::WorldObject>{};
			};
		};
	}
}

#endif //CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP
