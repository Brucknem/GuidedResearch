//
// Created by brucknem on 04.02.21.
//

#include "Residuals.hpp"

#include <utility>
#include "glog/logging.h"

namespace providentia {
	namespace calibration {

#pragma region CorrespondenceResidualBase

		CorrespondenceResidualBase::CorrespondenceResidualBase(Eigen::Vector2d _expectedPixel,
															   Eigen::Vector2d _frustumParameters,
															   Eigen::Vector3d _intrinsics,
															   Eigen::Vector2d _imageSize) :
				expectedPixel(std::move(_expectedPixel)),
				frustumParameters(std::move(_frustumParameters)),
				intrinsics(std::move(_intrinsics)),
				imageSize(std::move(_imageSize)) {}

		template<typename T>
		bool CorrespondenceResidualBase::calculateResidual(const T *_translation, const T *_rotation,
														   const T *_point, T *residual) const {
			Eigen::Matrix<T, 2, 1> _frustumParameters{(T) frustumParameters.x(), (T) frustumParameters.y()};
			Eigen::Matrix<T, 3, 1> _intrinsics{(T) intrinsics.x(), (T) intrinsics.y(), (T) intrinsics.z()};
			Eigen::Matrix<T, 2, 1> _imageSize{(T) imageSize.x(), (T) imageSize.y()};
			Eigen::Matrix<T, 4, 1> point{_point[0], _point[1], _point[2], (T) 1};

			Eigen::Matrix<T, 2, 1> actualPixel;
			actualPixel = camera::render(_translation, _rotation, _frustumParameters.data(),
										 _intrinsics.data(), _imageSize.data(), point.data());

			residual[0] = expectedPixel.x() - actualPixel.x();
			residual[1] = expectedPixel.y() - actualPixel.y();
			return true;
		}

#pragma endregion CorrespondenceResidualBase

#pragma region PointCorrespondenceResidual

		PointCorrespondenceResidual::PointCorrespondenceResidual(Eigen::Vector2d _expectedPixel,
																 Eigen::Vector3d _worldPosition,
																 Eigen::Vector2d _frustumParameters,
																 Eigen::Vector3d _intrinsics,
																 Eigen::Vector2d _imageSize) :
				CorrespondenceResidualBase(
						std::move(_expectedPixel),
						std::move(_frustumParameters),
						std::move(_intrinsics),
						std::move(_imageSize)
				),
				worldPosition(std::move(_worldPosition)) {}

		template<typename T>
		bool PointCorrespondenceResidual::operator()(const T *_translation, const T *_rotation, T *residual) const {
			Eigen::Matrix<T, 3, 1> point = worldPosition.cast<T>();
			return calculateResidual(_translation, _rotation, point.data(), residual);
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

#pragma endregion PointCorrespondenceResidual

#pragma region LineCorrespondenceResidual

		LineCorrespondenceResidual::LineCorrespondenceResidual(Eigen::Vector2d _expectedPixel,
															   Eigen::Vector3d _lineOrigin,
															   const Eigen::Vector3d &_lineHeading,
															   Eigen::Vector2d _frustumParameters,
															   Eigen::Vector3d _intrinsics,
															   Eigen::Vector2d _imageSize) :
				CorrespondenceResidualBase(
						std::move(_expectedPixel),
						std::move(_frustumParameters),
						std::move(_intrinsics),
						std::move(_imageSize)
				),
				lineOrigin(std::move(_lineOrigin)),
				lineHeading(_lineHeading.normalized()) {}

		template<typename T>
		bool LineCorrespondenceResidual::operator()(const T *_translation, const T *_rotation, const T *_lambda,
													T *residual) const {
			Eigen::Matrix<T, 3, 1> point = lineOrigin.cast<T>();
			point += lineHeading.cast<T>() * _lambda[0];
			return calculateResidual(_translation, _rotation, point.data(), residual);
		}

		ceres::CostFunction *
		LineCorrespondenceResidual::Create(Eigen::Vector2d _expectedPixel,
										   Eigen::Vector3d _lineOrigin,
										   Eigen::Vector3d _lineHeading,
										   Eigen::Vector2d _frustumParameters,
										   Eigen::Vector3d _intrinsics,
										   Eigen::Vector2d _imageSize) {
			return new ceres::AutoDiffCostFunction<LineCorrespondenceResidual, 2, 3, 3, 1>(
					new LineCorrespondenceResidual(
							std::move(_expectedPixel), std::move(_lineOrigin), std::move(_lineHeading),
							std::move(_frustumParameters), std::move(_intrinsics),
							std::move(_imageSize)
					)
			);
		}

#pragma endregion LineCorrespondenceResidual

#pragma region PlaneCorrespondenceResidual

		PlaneCorrespondenceResidual::PlaneCorrespondenceResidual(Eigen::Vector2d _expectedPixel,
																 Eigen::Vector3d _planeOrigin,
																 const Eigen::Vector3d &_planeSideA,
																 const Eigen::Vector3d &_planeSideB,
																 Eigen::Vector2d _frustumParameters,
																 Eigen::Vector3d _intrinsics,
																 Eigen::Vector2d _imageSize) :
				CorrespondenceResidualBase(
						std::move(_expectedPixel),
						std::move(_frustumParameters),
						std::move(_intrinsics),
						std::move(_imageSize)
				),
				planeOrigin(std::move(_planeOrigin)),
				planeSideA(_planeSideA.normalized()),
				planeSideB(_planeSideB.normalized()) {}

		template<typename T>
		bool PlaneCorrespondenceResidual::operator()(const T *_translation, const T *_rotation, const T *_lambda,
													 const T *_mu, T *residual) const {
			Eigen::Matrix<T, 3, 1> point = planeOrigin.cast<T>();
			point += planeSideA.cast<T>() * _lambda[0];
			point += planeSideB.cast<T>() * _mu[0];
			return calculateResidual(_translation, _rotation, point.data(), residual);
		}

		ceres::CostFunction *
		PlaneCorrespondenceResidual::Create(Eigen::Vector2d _expectedPixel,
											Eigen::Vector3d _planeOrigin,
											Eigen::Vector3d _planeSideA,
											Eigen::Vector3d _planeSideB,
											Eigen::Vector2d _frustumParameters,
											Eigen::Vector3d _intrinsics,
											Eigen::Vector2d _imageSize) {
			return new ceres::AutoDiffCostFunction<PlaneCorrespondenceResidual, 2, 3, 3, 1, 1>(
					new PlaneCorrespondenceResidual(
							std::move(_expectedPixel), std::move(_planeOrigin), std::move(_planeSideA), std::move
									(_planeSideB),
							std::move(_frustumParameters), std::move(_intrinsics),
							std::move(_imageSize)
					)
			);
		}

#pragma endregion PlaneCorrespondenceResidual

	}
}
