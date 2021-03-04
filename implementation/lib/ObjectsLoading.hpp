//
// Created by brucknem on 04.03.21.
//

#ifndef CAMERASTABILIZATION_OBJECTSLOADING_HPP
#define CAMERASTABILIZATION_OBJECTSLOADING_HPP

#include <string>
#include "yaml-cpp/yaml.h"

#include "WorldObjects.hpp"

namespace YAML {
	template<>
	struct convert<Eigen::Vector3d> {

		static bool decode(const Node &node, Eigen::Vector3d &rhs) {
			if (!node.IsSequence() || node.size() != 3) {
				return false;
			}

			rhs.x() = node[0].as<double>();
			rhs.y() = node[1].as<double>();
			rhs.z() = node[2].as<double>();
			return true;
		}
	};

	template<>
	struct convert<Eigen::Vector2d> {

		static bool decode(const Node &node, Eigen::Vector2d &rhs) {
			if (!node.IsSequence() || node.size() != 2) {
				return false;
			}

			rhs.x() = node[0].as<double>();
			rhs.y() = node[1].as<double>();
			return true;
		}
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
		YAML::Node LoadYAML(const std::string &filename) {
			int l = filename.length();
			if (
				filename.at(l - 5) != '.' ||
				filename.at(l - 4) != 'y' ||
				filename.at(l - 3) != 'a' ||
				filename.at(l - 2) != 'm' ||
				filename.at(l - 1) != 'l'
				) {
				throw std::invalid_argument(filename + " is not a YAML file.");
			}
			return YAML::LoadFile(filename);
		}

		/**
		 * Loads the world positions of the objects from a YAML object.
		 *
		 * @param yaml
		 * @return
		 */
		std::vector<ParametricPoint> LoadObjects(YAML::Node yaml) {
			std::vector<ParametricPoint> objects;
			assert(yaml["objects"].IsSequence());

			for (const auto &object : yaml["objects"]) {
				if (std::strcmp(object["type"].as<std::string>().c_str(), "pole") == 0 &&
					std::strcmp(object["name"].as<std::string>().c_str(), "permanentDelineator") == 0
					) {
					const Eigen::Vector3d &worldPosition = object["position"].as<Eigen::Vector3d>();

					bool hasPixels = false;
					for (const auto &pixelNode : object["pixels"]) {
						const Eigen::Vector2d &pixel = pixelNode.as<Eigen::Vector2d>();
						objects.emplace_back(ParametricPoint::OnPoint(pixel, worldPosition));
						hasPixels = true;
					}

					if (!hasPixels) {
						objects.emplace_back(ParametricPoint::OnPoint(worldPosition));
					}
				}
			}
			return objects;
		}

		/**
		 * Loads the world positions of the objects from a YAML file.
		 *
		 * @param filename
		 * @return
		 */
		std::vector<ParametricPoint> LoadObjects(const std::string &filename) {
			YAML::Node yaml = LoadYAML(filename);
			return LoadObjects(yaml);
		}
	}
}

#endif //CAMERASTABILIZATION_OBJECTSLOADING_HPP
