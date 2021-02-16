//
// Created by brucknem on 16.02.21.
//

#include "HDMap.hpp"

#include <utility>

namespace providentia {
	namespace calibration {
		HDMap::HDMap(std::string _filename) : filename(std::move(_filename)) {
			pugi::xml_parse_result result = doc.load_file(filename.c_str());

			if (!result) {
				// XML file is not ok ... we throw some exception
				throw std::invalid_argument("XML file parsing failed: " + std::string(result.description()));
			} // if
		}

		std::vector<pugi::xml_node> HDMap::findNodesByType(const std::string &type) {
			return findNodesByXPath("//" + type);
		}

		std::vector<pugi::xml_node> HDMap::findNodesByXPath(const std::string &path) {
			std::vector<pugi::xml_node> nodes;
			pugi::xpath_node_set nodeSet = doc.select_nodes((path).c_str());

			for (pugi::xpath_node node : nodeSet) {
				nodes.emplace_back(node.node());
			}

			return nodes;
		}

		pugi::xml_object_range<pugi::xml_named_node_iterator> HDMap::getRoads() {
			return doc.child("OpenDRIVE").children("road");
		}

		pugi::xml_node HDMap::getHeader() {
			return doc.child("OpenDRIVE").child("header");
		}

		std::string HDMap::getHeader(const std::string &attribute) {
			return getHeader().attribute(attribute.c_str()).as_string();
		}

		std::string HDMap::getGeoReference() {
			return getHeader().child("geoReference").child_value();
		}
	}
}
