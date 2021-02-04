//
// Created by brucknem on 02.02.21.
//

#include "CameraPoseEstimation.hpp"

#include <utility>
#include "ceres/autodiff_cost_function.h"

namespace providentia {
	namespace calibration {

		void CameraPoseEstimator::estimate(bool _logSummary) {
			options.linear_solver_type = ceres::DENSE_QR;
			options.minimizer_progress_to_stdout = _logSummary;
			Solve(options, &problem, &summary);
		}

		void CameraPoseEstimator::addPointCorrespondence(const Eigen::Vector3d &worldPosition,
														 const Eigen::Vector2d &pixel) {
			problem.AddResidualBlock(
					PointCorrespondenceResidual::Create(pixel, worldPosition, frustumParameters, intrinsics, imageSize),
					nullptr, translation.data(), rotation.data()
			);
		}

		CameraPoseEstimator::CameraPoseEstimator(const Eigen::Vector3d &_initialTranslation,
												 const Eigen::Vector3d &_initialRotation,
												 Eigen::Vector2d _frustumParameters,
												 Eigen::Vector3d _intrinsics,
												 Eigen::Vector2d _imageSize) :
				initialRotation(_initialRotation), initialTranslation(_initialTranslation),
				rotation(_initialRotation), translation(_initialTranslation),
				frustumParameters(std::move(_frustumParameters)), intrinsics(std::move(_intrinsics)),
				imageSize(std::move(_imageSize)) {}

		const Eigen::Vector3d &CameraPoseEstimator::getTranslation() const {
			return translation;
		}

		const Eigen::Vector3d &CameraPoseEstimator::getRotation() const {
			return rotation;
		}

	}
}
