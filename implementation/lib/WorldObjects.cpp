//
// Created by brucknem on 08.02.21.
//

#include "WorldObjects.hpp"

#include <utility>

namespace providentia {
	namespace calibration {
#pragma region Points

		ParametricPoint::ParametricPoint(Eigen::Vector3d origin, const Eigen::Vector3d
		&axisA, const Eigen::Vector3d &axisB, double lambda, double mu) :
			expectedPixel({0, 0}), origin(std::move(origin)), axisA(axisA.normalized()),
			axisB(axisB.normalized()),
			lambda(new double(lambda)), mu(new double(mu)) {}

		Eigen::Vector3d ParametricPoint::getPosition() const {
			return origin + *lambda * axisA + *mu * axisB;
		}

		const Eigen::Vector3d &ParametricPoint::getOrigin() const {
			return origin;
		}

		const Eigen::Vector3d &ParametricPoint::getAxisA() const {
			return axisA;
		}

		const Eigen::Vector3d &ParametricPoint::getAxisB() const {
			return axisB;
		}

		double *ParametricPoint::getLambda() const {
			return lambda;
		}

		double *ParametricPoint::getMu() const {
			return mu;
		}

		const Eigen::Vector2d &ParametricPoint::getExpectedPixel() const {
			return expectedPixel;
		}

		void ParametricPoint::setExpectedPixel(const Eigen::Vector2d &value) {
			expectedPixel = value;
			isExpectedPixelSet = true;
		}

		ParametricPoint ParametricPoint::onPlane(const Eigen::Vector2d &expectedPixel, Eigen::Vector3d origin,
												 const Eigen::Vector3d &axisA, const Eigen::Vector3d &axisB,
												 double lambda, double mu) {
			ParametricPoint point = onPlane(std::move(origin), axisA, axisB, lambda, mu);
			point.setExpectedPixel(expectedPixel);
			return point;
		}

		ParametricPoint ParametricPoint::onPlane(Eigen::Vector3d origin, const Eigen::Vector3d &axisA,
												 const Eigen::Vector3d &axisB, double lambda, double mu) {
			return ParametricPoint(std::move(origin), axisA.stableNormalized(), axisB.stableNormalized(), lambda,
								   mu);
		}

		ParametricPoint ParametricPoint::onLine(const Eigen::Vector2d &expectedPixel, Eigen::Vector3d origin, const
		Eigen::Vector3d &heading, double lambda) {
			ParametricPoint point = onLine(std::move(origin), heading, lambda);
			point.setExpectedPixel(expectedPixel);
			return point;
		}

		ParametricPoint ParametricPoint::onLine(Eigen::Vector3d origin, const Eigen::Vector3d &heading,
												double lambda) {
			return ParametricPoint::onPlane(std::move(origin), heading, {0, 0, 0}, lambda, 0);
		}

		ParametricPoint
		ParametricPoint::onPoint(const Eigen::Vector2d &expectedPixel, const Eigen::Vector3d &worldPosition) {
			ParametricPoint point = onPoint(worldPosition);
			point.setExpectedPixel(expectedPixel);
			return point;
		}

		ParametricPoint ParametricPoint::onPoint(const Eigen::Vector3d &worldPosition) {
			return ParametricPoint::onLine(worldPosition, {0, 0, 0}, 0);
		}

		bool ParametricPoint::hasExpectedPixel() const {
			return isExpectedPixelSet;
		}

#pragma endregion Points

#pragma region Objects

		void WorldObject::add(const ParametricPoint &point) {
			points.emplace_back(point);
		}

		double WorldObject::getWeight() const {
			if (points.empty()) {
				return 0;
			}
			return 1. / (double) points.size();
		}

		const std::vector<ParametricPoint> &WorldObject::getPoints() const {
			return points;
		}

		WorldObject::WorldObject(const ParametricPoint &point) {
			add(point);
		}

		Eigen::Vector3d WorldObject::getMean() const {
			Eigen::Vector3d mean{0, 0, 0};
			for (const auto &point : points) {
				mean += point.getPosition();
			}
			return mean / points.size();
		}

		const std::string &WorldObject::getId() const {
			return id;
		}

		void WorldObject::setId(const std::string &value) {
			id = value;
		}

		double WorldObject::getHeight() const {
			return height;
		}

		void WorldObject::setHeight(double value) {
			height = value;
		}

		int WorldObject::getNumPoints() const {
			return (int) points.size();
		}

#pragma endregion Objects
	}
}