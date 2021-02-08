//
// Created by brucknem on 04.02.21.
//

#include "Residuals.hpp"

#include <utility>
#include "glog/logging.h"

namespace providentia {
	namespace calibration {

#pragma region CorrespondenceResidualBase

		CorrespondenceResidual::CorrespondenceResidual(Eigen::Vector2d _expectedPixel,
													   std::shared_ptr<providentia::calibration::PointOnPlane> _plane,
													   Eigen::Vector2d _frustumParameters,
													   Eigen::Vector3d _intrinsics,
													   Eigen::Vector2d _imageSize) :
				expectedPixel(std::move(_expectedPixel)),
				plane(std::move(_plane)),
				frustumParameters(std::move(_frustumParameters)),
				intrinsics(std::move(_intrinsics)),
				imageSize(std::move(_imageSize)) {}

		template<typename T>
		bool CorrespondenceResidual::calculateResidual(const T *_translation, const T *_rotation,
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

		template<typename T>
		bool CorrespondenceResidual::operator()(const T *_translation, const T *_rotation, const T *_lambda,
												const T *_mu, T *residual) const {
			Eigen::Matrix<T, 3, 1> point = plane->getOrigin().cast<T>();
			point += plane->getAxisA().cast<T>() * _lambda[0];
			point += plane->getAxisB().cast<T>() * _mu[0];
			return calculateResidual(_translation, _rotation, point.data(), residual);
		}

		ceres::CostFunction *
		CorrespondenceResidual::Create(const Eigen::Vector2d &_expectedPixel,
									   const std::shared_ptr<providentia::calibration::PointOnPlane> &_plane,
									   const Eigen::Vector2d &_frustumParameters,
									   const Eigen::Vector3d &_intrinsics,
									   const Eigen::Vector2d &_imageSize) {
			return new ceres::AutoDiffCostFunction<CorrespondenceResidual, 2, 3, 3, 1, 1>(
					new CorrespondenceResidual(_expectedPixel, _plane, _frustumParameters, _intrinsics, _imageSize)
			);
		}

#pragma endregion CorrespondenceResidualBase
	}
}
