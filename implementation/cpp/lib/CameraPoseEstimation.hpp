//
// Created by brucknem on 02.02.21.
//

#ifndef CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP
#define CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP

#include <utility>
#include <vector>
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"

#include "Residuals.hpp"
#include "WorldObjects.hpp"

namespace providentia {
	namespace calibration {
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
			 * The [near, far] plane distances of the view frustum.
			 */
			Eigen::Vector2d frustumParameters;

			/**
			 * The [sensorWidth, aspectRatio, focalLength] of the pinhole camera model.
			 */
			Eigen::Vector3d intrinsics;

			/**
			 * The [width, height] of the image.
			 */
			Eigen::Vector2d imageSize;

			/**
			 * A buffer for the known world worldObjects.
			 */
			std::vector<providentia::calibration::WorldObject> worldObjects;

			bool hasInitialGuessSet = false;

		public:
			/**
			 * @constructor
			 *
			 * @param _frustumParameters The [near, far] plane distances of the view frustum.
			 * @param _intrinsics The [sensorWidth, aspectRatio, focalLength] of the pinhole camera model.
			 * @param _imageSize The [width, height] of the image.
			 */
			CameraPoseEstimator(Eigen::Vector2d _frustumParameters,
								Eigen::Vector3d _intrinsics,
								Eigen::Vector2d _imageSize);

			/**
			 * @destructor
			 */
			virtual ~CameraPoseEstimator() = default;

			void addWorldObject(const WorldObject &worldObject);

			/**
			 * Estimates the camera translation and rotation based on the known correspondences between the world and
			 * image.
			 */
			void estimate(bool _logSummary = false);

			/**
			 * Based on the known world positions calculates and initial guess for the camera translation and rotation.
			 * This is necessary as the optimization problem is rather ill posed and sensitive to the initialization.
			 */
			void calculateInitialGuess();

			void setInitialGuess(Eigen::Vector3d translation, Eigen::Vector3d rotation);

			/**
			 * @get
			 */
			const Eigen::Vector3d &getTranslation() const;

			/**
			 * @get
			 */
			const Eigen::Vector3d &getRotation() const;

			const std::vector<providentia::calibration::WorldObject> &getWorldObjects() const;

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
		};
	}
}

#endif //CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP
