//
// Created by brucknem on 08.02.21.
//

#include "WorldObjects.hpp"

#include <utility>

namespace providentia {
	namespace calibration {
		PointOnPlane::PointOnPlane(Eigen::Vector3d _origin, const Eigen::Vector3d &_axisA,
								   const Eigen::Vector3d &_axisB, double
								   _lambda, double _mu) :
				origin(std::move(_origin)), axisA(_axisA.normalized()), axisB(_axisB.normalized()),
				lambda(_lambda), mu(_mu) {}

		Eigen::Vector3d PointOnPlane::getPosition() const {
			return origin + lambda * axisA + mu * axisB;
		}

		const Eigen::Vector3d &PointOnPlane::getOrigin() const {
			return origin;
		}

		const Eigen::Vector3d &PointOnPlane::getAxisA() const {
			return axisA;
		}

		const Eigen::Vector3d &PointOnPlane::getAxisB() const {
			return axisB;
		}

		const double *PointOnPlane::getLambda() const {
			return &lambda;
		}

		const double *PointOnPlane::getMu() const {
			return &mu;
		}

		PointOnLine::PointOnLine(Eigen::Vector3d _origin, const Eigen::Vector3d &_heading, double _lambda) :
				PointOnPlane(std::move(_origin), _heading, {0, 0, 0}, _lambda) {}

		Point::Point(Eigen::Vector3d _worldPosition) :
				PointOnLine(std::move(_worldPosition), {0, 0, 0}) {}

	}
}