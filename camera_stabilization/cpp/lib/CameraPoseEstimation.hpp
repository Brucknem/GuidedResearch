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
			 * A buffer for the known world positions used to calculate an initial guess for the optimization.
			 */
			std::vector<Eigen::Vector3d> worldPositions;

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

			/**
			 * Adds a correspondence to the optimization problem.
			 *
			 * @param worldPosition The [x, y, z] world position of the correspondence.
			 * @param pixel The [u, v] pixel position of the correspondence.
			 */
			void addPointCorrespondence(const Eigen::Vector3d &worldPosition, const Eigen::Vector2d &pixel);

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

			/**
			 * @get
			 */
			const Eigen::Vector3d &getTranslation() const;

			/**
			 * @get
			 */
			const Eigen::Vector3d &getRotation() const;

			Eigen::Vector3d calculateMean();

			Eigen::Vector3d calculateFurthestKnownWorldPosition(const Eigen::Vector3d &mean);
		};
	}
}

#endif //CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP
