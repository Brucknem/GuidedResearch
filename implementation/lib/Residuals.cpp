//
// Created by brucknem on 04.02.21.
//

#include "Residuals.hpp"

#include <utility>
#include "glog/logging.h"

namespace providentia {
	namespace calibration {

#pragma region DistanceResidual

		DistanceResidual::DistanceResidual(double expectedValue) : expectedValue(expectedValue) {}

		template<typename T>
		bool DistanceResidual::operator()(const T *value, T *residual) const {
			residual[0] = value[0] - (T) expectedValue;
			return true;
		}

		ceres::CostFunction *DistanceResidual::Create(const double expectedValue) {
			return new ceres::AutoDiffCostFunction<DistanceResidual, 1, 1>(
				new DistanceResidual(expectedValue)
			);
		}

#pragma endregion DistanceResidual

#pragma region DistanceFromIntervalResidual

		DistanceFromIntervalResidual::DistanceFromIntervalResidual(double upperBound) : DistanceFromIntervalResidual
																							(0, upperBound) {}

		DistanceFromIntervalResidual::DistanceFromIntervalResidual(double lowerBound, double upperBound) : lowerBound(
			lowerBound), upperBound(upperBound) {}

		template<typename T>
		bool DistanceFromIntervalResidual::operator()(const T *value, T *residual) const {
			if (value[0] > (T) upperBound) {
				residual[0] = value[0] - (T) upperBound;
			} else if (value[0] < (T) lowerBound) {
				residual[0] = value[0] - (T) lowerBound;
			} else {
				residual[0] = (T) 0;
			}
			return true;
		}

		ceres::CostFunction *DistanceFromIntervalResidual::Create(double lowerBound, double upperBound) {
			return new ceres::AutoDiffCostFunction<DistanceFromIntervalResidual, 1, 1>(
				new DistanceFromIntervalResidual(lowerBound, upperBound)
			);
		}

		ceres::CostFunction *DistanceFromIntervalResidual::Create(const double upperBound) {
			return new ceres::AutoDiffCostFunction<DistanceFromIntervalResidual, 1, 1>(
				new DistanceFromIntervalResidual(0, upperBound)
			);
		}

#pragma endregion DistanceFromIntervalResidual

#pragma region CorrespondenceResidualBase

		CorrespondenceResidual::CorrespondenceResidual(const Eigen::Vector2d &_expectedPixel,
													   const providentia::calibration::ParametricPoint &_point,
													   const Eigen::Matrix<double, 3, 4> &_intrinsics) :
			expectedPixel(_expectedPixel),
			parametricPoint(_point),
			intrinsics(_intrinsics) {}

		template<typename T>
		bool CorrespondenceResidual::operator()(
			const T *tx,
			const T *ty,
			const T *tz,
			const T *rx,
			const T *ry,
			const T *rz,
			const T *_lambda,
			const T *_mu,
			const T *_weight,
			T *residual) const {
			Eigen::Matrix<T, 3, 1> point = parametricPoint.getOrigin().cast<T>();
			point += parametricPoint.getAxisA().cast<T>() * _lambda[0];
			point += parametricPoint.getAxisB().cast<T>() * _mu[0];

			Eigen::Matrix<T, 2, 1> actualPixel;
			bool flipped;
			actualPixel = camera::render(
				new T[3]{tx[0], ty[0], tz[0]},
				new T[3]{rx[0], ry[0], rz[0]},
				intrinsics,
				point.data(),
				flipped
			);

			residual[0] = expectedPixel.x() - actualPixel.x();
			residual[1] = expectedPixel.y() - actualPixel.y();

			residual[0] = residual[0] * _weight[0];
			residual[1] = residual[1] * _weight[0];

			return !flipped;
		}

		ceres::CostFunction *
		CorrespondenceResidual::Create(const Eigen::Vector2d &_expectedPixel,
									   const providentia::calibration::ParametricPoint &_point,
									   const Eigen::Matrix<double, 3, 4> &_intrinsics) {
			return new ceres::AutoDiffCostFunction<CorrespondenceResidual, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
				new CorrespondenceResidual(_expectedPixel, _point, _intrinsics),
				ceres::TAKE_OWNERSHIP
			);
		}

#pragma endregion CorrespondenceResidualBase

		template bool DistanceFromIntervalResidual::operator()(const ceres::Jet<double, 1> *, ceres::Jet<double, 1>
		*) const;

		template bool DistanceFromIntervalResidual::operator()(const double *, double *) const;

		template bool DistanceResidual::operator()(const ceres::Jet<double, 1> *, ceres::Jet<double, 1> *) const;

		template bool DistanceResidual::operator()(const double *, double *) const;

	}
}
