//
// Created by brucknem on 04.03.21.
//

#include "ObjectsLoading.hpp"

namespace YAML {

	bool convert<Eigen::Vector3d>::decode(const Node &node, Eigen::Vector3d &rhs) {
		if (!node.IsSequence() || node.size() != 3) {
			return false;
		}

		rhs.x() = node[0].as<double>();
		rhs.y() = node[1].as<double>();
		rhs.z() = node[2].as<double>();
		return true;
	}

	bool convert<Eigen::Vector2d>::decode(const Node &node, Eigen::Vector2d &rhs) {
		if (!node.IsSequence() || node.size() != 2) {
			return false;
		}

		rhs.x() = node[0].as<double>();
		rhs.y() = node[1].as<double>();
		return true;
	}
}

namespace providentia {
	namespace calibration {
		YAML::Node loadYAML(const std::string &filename) {
			size_t l = filename.length();
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

		std::vector<WorldObject>
		loadObjects(YAML::Node opendriveObjects, const YAML::Node &imageObjects, Eigen::Vector2i imageSize) {
			std::vector<WorldObject> objects;
			assert(opendriveObjects["objects"].IsSequence());

			auto imageHeight = imageSize[1];

			for (const auto &object : opendriveObjects["objects"]) {
				WorldObject worldObject;

				std::string objectId = object["id"].as<std::string>();
				worldObject.setId(objectId);
				worldObject.setHeight(object["height"].as<double>());

				if (std::strcmp(object["type"].as<std::string>().c_str(), "pole") == 0 &&
					std::strcmp(object["name"].as<std::string>().c_str(), "permanentDelineator") == 0) {
					const Eigen::Vector3d &worldPosition = object["position"].as<Eigen::Vector3d>();
//					auto heading = object["hdg"].as<double>();

					bool hasPixels = false;
					for (const auto imageObject : imageObjects) {
						std::string frameObjectId = imageObject["id"].as<std::string>();
						if (std::strcmp(frameObjectId.c_str(), objectId.c_str()) == 0) {
							for (const auto pixelNode : imageObject["pixels"]) {
								Eigen::Vector2d pixel = pixelNode.as<Eigen::Vector2d>();
								if (imageHeight > 1) {
									pixel = {pixel.x(), imageHeight - 1 - pixel.y()};
								}
								worldObject.add(
									ParametricPoint::onLine(pixel, worldPosition, Eigen::Vector3d::UnitZ()));
								hasPixels = true;
							}
						}
					}

					if (!hasPixels) {
						worldObject.add(ParametricPoint::onPoint(worldPosition));
					}
					objects.emplace_back(worldObject);
				}
			}
			return objects;
		}

		std::vector<WorldObject>
		loadObjects(const std::string &opendriveObjectsFile, const std::string &imageObjectsFile,
					Eigen::Vector2i imageSize) {
			YAML::Node opendriveObjects = loadYAML(opendriveObjectsFile);
			YAML::Node imageObjects = loadYAML(imageObjectsFile);
			return loadObjects(opendriveObjects, imageObjects, std::move(imageSize));
		}
	}
}