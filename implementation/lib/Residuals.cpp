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
													   Eigen::Matrix<double, 3, 4> _intrinsics,
													   double _weight) :
			expectedPixel(std::move(_expectedPixel)),
			plane(std::move(_point)),
			intrinsics(std::move(_intrinsics)),
			weight(_weight) {}

		template<typename T>
		bool CorrespondenceResidual::calculateResidual(const T *_translation, const T *_rotation,
													   const T *_point, T *residual) const {
			Eigen::Matrix<T, 4, 1> point{_point[0], _point[1], _point[2], (T) 1};

//			std::cout << "Translation" << std::endl;
//			std::cout << _translation[0] << ", " << _translation[1] << ", " << _translation[2] << std::endl
//					  << std::endl;
//			std::cout << "Rotation" << std::endl;
//			std::cout << _rotation[0] << ", " << _rotation[1] << ", " << _rotation[2] << std::endl << std::endl;

			Eigen::Matrix<T, 2, 1> actualPixel;
			bool flipped;
			actualPixel = camera::render(_translation, _rotation, intrinsics, point.data(), flipped);

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
			bool result = calculateResidual(_translation, _rotation, point.data(), residual);

			residual[2] = _lambda[0] * weight;
			residual[3] = _mu[0] * weight;
			return result;
		}

		ceres::CostFunction *
		CorrespondenceResidual::Create(const Eigen::Vector2d &_expectedPixel,
									   const std::shared_ptr<providentia::calibration::ParametricPoint> &_plane,
									   Eigen::Matrix<double, 3, 4> _intrinsics,
									   double _weight) {
			return new ceres::AutoDiffCostFunction<CorrespondenceResidual, 4, 3, 3, 1, 1>(
				new CorrespondenceResidual(_expectedPixel, _plane, std::move(_intrinsics),
										   _weight)
			);
		}

#pragma endregion CorrespondenceResidualBase
	}
}
