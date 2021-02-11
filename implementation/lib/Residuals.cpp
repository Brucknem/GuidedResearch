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
													   std::shared_ptr<providentia::calibration::ParametricPoint> _point,
													   Eigen::Vector2d _frustumParameters,
													   Eigen::Vector3d _intrinsics,
													   Eigen::Vector2d _imageSize,
													   double _weight) :
			expectedPixel(std::move(_expectedPixel)),
			plane(std::move(_point)),
			frustumParameters(std::move(_frustumParameters)),
			intrinsics(std::move(_intrinsics)),
			imageSize(std::move(_imageSize)),
			weight(_weight) {}

		template<typename T>
		bool CorrespondenceResidual::calculateResidual(const T *_translation, const T *_rotation,
													   const T *_point, T *residual) const {
			Eigen::Matrix<T, 2, 1> _frustumParameters{(T) frustumParameters.x(), (T) frustumParameters.y()};
			Eigen::Matrix<T, 3, 1> _intrinsics{(T) intrinsics.x(), (T) intrinsics.y(), (T) intrinsics.z()};
			Eigen::Matrix<T, 2, 1> _imageSize{(T) imageSize.x(), (T) imageSize.y()};
			Eigen::Matrix<T, 4, 1> point{_point[0], _point[1], _point[2], (T) 1};

//			std::cout << "Translation" << std::endl;
//			std::cout << _translation[0] << ", " << _translation[1] << ", " << _translation[2] << std::endl
//					  << std::endl;
//			std::cout << "Rotation" << std::endl;
//			std::cout << _rotation[0] << ", " << _rotation[1] << ", " << _rotation[2] << std::endl << std::endl;

			Eigen::Matrix<T, 2, 1> actualPixel;
			actualPixel = camera::render(_translation, _rotation, _frustumParameters.data(),
										 _intrinsics.data(), _imageSize.data(), point.data());

			residual[0] = expectedPixel.x() - actualPixel.x();
			residual[1] = expectedPixel.y() - actualPixel.y();

			residual[0] *= (T) weight;
			residual[1] *= (T) weight;

//			LOG(INFO) << std::endl << residual[0] << std::endl << residual[1];
			return true;
		}

		template<typename T>
		bool CorrespondenceResidual::operator()(const T *_translation, const T *_rotation, const T *_lambda,
												const T *_mu, T *residual) const {
			Eigen::Matrix<T, 3, 1> point = plane->getOrigin().cast<T>();
//			std::cout << "Point" << std::endl;
//			std::cout << point << std::endl << std::endl;
			point += plane->getAxisA().cast<T>() * _lambda[0];
//			std::cout << point << std::endl << std::endl;
			point += plane->getAxisB().cast<T>() * _mu[0];
//			std::cout << point << std::endl << std::endl;
			return calculateResidual(_translation, _rotation, point.data(), residual);
		}

		ceres::CostFunction *
		CorrespondenceResidual::Create(const Eigen::Vector2d &_expectedPixel,
									   const std::shared_ptr<providentia::calibration::ParametricPoint> &_plane,
									   const Eigen::Vector2d &_frustumParameters,
									   const Eigen::Vector3d &_intrinsics,
									   const Eigen::Vector2d &_imageSize,
									   double _weight) {
			return new ceres::AutoDiffCostFunction<CorrespondenceResidual, 2, 3, 3, 1, 1>(
				new CorrespondenceResidual(_expectedPixel, _plane, _frustumParameters, _intrinsics, _imageSize,
										   _weight)
			);
		}

#pragma endregion CorrespondenceResidualBase
	}
}
