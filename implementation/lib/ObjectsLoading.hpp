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
		 * @param opendriveObjects
		 * @return
		 */
		std::vector<WorldObject>
		LoadObjects(YAML::Node opendriveObjects, const YAML::Node &imageObjects, Eigen::Vector2i
		imageSize = {-1, -1}) {
			std::vector<WorldObject> objects;
			assert(opendriveObjects["objects"].IsSequence());

			auto imageHeight = imageSize[1];

			for (const auto &object : opendriveObjects["objects"]) {
//				if (object["id"].as<int>() != 4007962) {
//					continue;
//				}

				WorldObject worldObject;

				std::string objectId = object["id"].as<std::string>();
				worldObject.setId(objectId);

				if (std::strcmp(object["type"].as<std::string>().c_str(), "pole") == 0 &&
					std::strcmp(object["name"].as<std::string>().c_str(), "permanentDelineator") == 0) {
					const Eigen::Vector3d &worldPosition = object["position"].as<Eigen::Vector3d>();

					bool hasPixels = false;
					for (const auto imageObject : imageObjects) {
						std::string frameObjectId = imageObject["id"].as<std::string>();
						if (std::strcmp(frameObjectId.c_str(), objectId.c_str()) == 0) {
							for (const auto pixelNode : imageObject["pixels"]) {
								Eigen::Vector2d pixel = pixelNode.as<Eigen::Vector2d>();
								if (imageHeight > 1) {
									pixel = {pixel.x(), imageHeight - 1 - pixel.y()};
								}
								worldObject.add(ParametricPoint::OnLine(pixel, worldPosition, {0, 0, 1}));
								hasPixels = true;
							}
						}
					}

					if (!hasPixels) {
						worldObject.add(ParametricPoint::OnPoint(worldPosition));
					}
					objects.emplace_back(worldObject);
				}
			}
			return objects;
		}

		/**
		 * Loads the world positions of the objects from a YAML file.
		 *
		 * @param opendriveObjectsFile
		 * @return
		 */
		std::vector<WorldObject> LoadObjects(const std::string &opendriveObjectsFile, const std::string
		&imageObjectsFile, Eigen::Vector2i imageSize = {-1, -1}) {
			YAML::Node opendriveObjects = LoadYAML(opendriveObjectsFile);
			YAML::Node imageObjects = LoadYAML(imageObjectsFile);
			return LoadObjects(opendriveObjects, imageObjects, std::move(imageSize));
		}
	}
}

#endif //CAMERASTABILIZATION_OBJECTSLOADING_HPP
