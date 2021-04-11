//
// Created by brucknem on 08.02.21.
//

#include "WorldObject.hpp"

#include <utility>

namespace providentia {
	namespace calibration {

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

	}
}