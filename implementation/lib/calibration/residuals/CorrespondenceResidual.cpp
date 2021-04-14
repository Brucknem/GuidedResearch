//
// Created by brucknem on 11.04.21.
//

#include "CorrespondenceResidual.hpp"
#include "../camera/RenderingPipeline.hpp"

namespace providentia {
	namespace calibration {
		namespace residuals {
			CorrespondenceResidual::CorrespondenceResidual(Eigen::Matrix<double, 2, 1> expectedPixel,
														   const ParametricPoint &point) :
				expectedPixel(std::move(expectedPixel)),
				parametricPoint(point) {}

			template<typename T>
			bool CorrespondenceResidual::operator()(
				const T *fx,
				const T *cx,
				const T *fy,
				const T *cy,
				const T *skew,
				const T *tx,
				const T *ty,
				const T *tz,
				const T *rx,
				const T *ry,
				const T *rz,
				const T *lambda,
				const T *mu,
				const T *weight,
				T *residual) const {
				Eigen::Matrix<T, 3, 1> point = parametricPoint.getOrigin().cast<T>();
				point += parametricPoint.getAxisA().cast<T>() * lambda[0];
				point += parametricPoint.getAxisB().cast<T>() * mu[0];

				Eigen::Matrix<T, 2, 1> actualPixel;
				bool flipped;
				actualPixel = providentia::camera::render(
					new T[3]{tx[0], ty[0], tz[0]},
					new T[3]{rx[0], ry[0], rz[0]},
					new T[5]{fx[0], cx[0], fy[0], cy[0], skew[0]},
					point.data(),
					flipped
				);

				residual[0] = expectedPixel.x() - actualPixel.x();
				residual[1] = expectedPixel.y() - actualPixel.y();

				residual[0] = residual[0] * weight[0];
				residual[1] = residual[1] * weight[0];

				return !flipped;
			}

			ceres::CostFunction *
			CorrespondenceResidual::create(const Eigen::Matrix<double, 2, 1> &expectedPixel,
										   const ParametricPoint &point) {
				return new ceres::AutoDiffCostFunction<CorrespondenceResidual, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
					new CorrespondenceResidual(expectedPixel, point),
					ceres::TAKE_OWNERSHIP
				);
			}
		}
	}
}