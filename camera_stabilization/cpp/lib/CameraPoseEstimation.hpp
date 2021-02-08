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
			 * A buffer for the known world positions.
			 */
			std::vector<providentia::calibration::Point> positions;

			/**
			 * A buffer for the known world lines.
			 */
			std::vector<providentia::calibration::PointOnLine> lines;

			/**
			 * A buffer for the known world planes.
			 */
			std::vector<providentia::calibration::PointOnPlane> planes;

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
			 * Adds a correspondence to the optimization problem.
			 *
			 * @param _origin The [x, y, z] origin in world position of the line that results in the expected pixel.
			 * @param _heading The [x, y, z] heading in world coordinates of the line that results in the expected pixel.
			 * @param pixel The [u, v] pixel position of the correspondence.
			 */
			void addLineCorrespondence(Eigen::Vector3d _origin, const Eigen::Vector3d &_heading,
									   const Eigen::Vector2d &pixel);

			/**
			 * Adds a correspondence to the optimization problem.
			 *
			 * @param _origin The [x, y, z] origin in world position of the plane that results in the expected
			 * pixel.
			 * @param _axisA The [x, y, z] axis in world coordinates of one side of the plane that results in
			 * the expected pixel.
			 * @param _axisB The [x, y, z] axis in world coordinates of another side of the plane that
			 * results in the expected pixel.
			 * @param pixel The [u, v] pixel position of the correspondence.
			 */
			void addPlaneCorrespondence(Eigen::Vector3d _origin, const Eigen::Vector3d &_axisA,
										const Eigen::Vector3d &_axisB, const Eigen::Vector2d &pixel);

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

			/**
			 * Calculates the mean of the known world correspondences.
			 */
			Eigen::Vector3d calculateMean();

			/**
			 * Adds a callback to the estimator that gets called after each optimization step.
			 */
			void addIterationCallback(ceres::IterationCallback *callback);
		};
	}
}

#endif //CAMERASTABILIZATION_CAMERAPOSEESTIMATION_HPP
