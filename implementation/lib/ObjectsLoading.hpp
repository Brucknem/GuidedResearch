//
// Created by brucknem on 04.03.21.
//

#ifndef CAMERASTABILIZATION_OBJECTSLOADING_HPP
#define CAMERASTABILIZATION_OBJECTSLOADING_HPP

#include <string>
#include <utility>
#include "yaml-cpp/yaml.h"

#include "WorldObjects.hpp"

namespace YAML {
	template<>
	struct convert<Eigen::Vector3d> {
		static bool decode(const Node &node, Eigen::Vector3d &rhs);
	};

	template<>
	struct convert<Eigen::Vector2d> {
		static bool decode(const Node &node, Eigen::Vector2d &rhs);
	};

}

namespace providentia {
	namespace calibration {

		/**
		 * Loads the given YAML file.
		 *
		 * @param filename
		 * @return
		 */
		YAML::Node loadYAML(const std::string &filename);

		/**
		 * Loads the world positions of the objects from a YAML object.
		 *
		 * @param opendriveObjects
		 * @return
		 */
		std::vector<WorldObject>
		loadObjects(YAML::Node opendriveObjects, const YAML::Node &imageObjects, Eigen::Vector2i
		imageSize = {-1, -1});

		/**
		 * Loads the world positions of the objects from a YAML file.
		 *
		 * @param opendriveObjectsFile
		 * @return
		 */
		std::vector<WorldObject> loadObjects(const std::string &opendriveObjectsFile, const std::string
		&imageObjectsFile, Eigen::Vector2i imageSize = {-1, -1});
	}
}

#endif //CAMERASTABILIZATION_OBJECTSLOADING_HPP
