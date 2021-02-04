//
// Created by brucknem on 04.02.21.
//

#include "Residuals.hpp"

#include <utility>

namespace providentia {
	namespace calibration {
		PointCorrespondenceResidual::PointCorrespondenceResidual(Eigen::Vector2d _expectedPixel,
																 Eigen::Vector3d _worldPosition,
																 Eigen::Vector2d _frustumParameters,
																 Eigen::Vector3d _intrinsics,
																 Eigen::Vector2d _imageSize) :
				expectedPixel(std::move(_expectedPixel)), worldPosition(std::move(_worldPosition)),
				frustumParameters(std::move(_frustumParameters)), intrinsics(std::move(_intrinsics)),
				imageSize(std::move(_imageSize)) {}

		template<typename T>
		bool PointCorrespondenceResidual::operator()(const T *_translation, const T *_rotation, T *residual) const {
			Eigen::Matrix<T, 4, 1> point{(T) worldPosition.x(), (T) worldPosition.y(), (T) worldPosition.z(),
										 (T) 1};

			Eigen::Matrix<T, 2, 1> _frustumParameters{(T) frustumParameters.x(), (T) frustumParameters.y()};
			Eigen::Matrix<T, 3, 1> _intrinsics{(T) intrinsics.x(), (T) intrinsics.y(), (T) intrinsics.z()};
			Eigen::Matrix<T, 2, 1> _imageSize{(T) imageSize.x(), (T) imageSize.y()};
			Eigen::Matrix<T, 2, 1> actualPixel;

			actualPixel = providentia::camera::render(_translation, _rotation, _frustumParameters.data(),
													  _intrinsics.data(), _imageSize.data(), point.data());

			residual[0] = expectedPixel.x() - actualPixel.x();
			residual[1] = expectedPixel.y() - actualPixel.y();

			return true;
		}

		ceres::CostFunction *
		PointCorrespondenceResidual::Create(Eigen::Vector2d _expectedPixel, Eigen::Vector3d _worldPosition,
											Eigen::Vector2d _frustumParameters, Eigen::Vector3d _intrinsics,
											Eigen::Vector2d _imageSize) {
			return new ceres::AutoDiffCostFunction<PointCorrespondenceResidual, 2, 3, 3>(
					new PointCorrespondenceResidual(
							std::move(_expectedPixel), std::move(_worldPosition),
							std::move(_frustumParameters), std::move(_intrinsics),
							std::move(_imageSize)
					)
			);
		}
	}
}
