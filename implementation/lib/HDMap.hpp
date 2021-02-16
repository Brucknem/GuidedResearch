//
// Created by brucknem on 16.02.21.
//

#ifndef CAMERASTABILIZATION_HDMAP_HPP
#define CAMERASTABILIZATION_HDMAP_HPP

#include <string>
#include <proj.h>
#include <exception>
#include <stdexcept>
#include <iostream>
#include <vector>

#include "pugixml.hpp"

namespace providentia {
	namespace calibration {

		/**
		 * A class for parsing and querying the OpenDrive HD maps.
		 */
		class HDMap {
		private:

			/**
			 * The filename of the OpenDrive HD map.
			 */
			std::string filename;

			/**
			 * The parsed XML document.
			 */
			pugi::xml_document doc;

		public:

			/**
			 * @constructor
			 */
			explicit HDMap(std::string filename);

			/**
			 * Finds all nodes by the given type.
			 */
			std::vector<pugi::xml_node> findNodesByType(const std::string &type);

			/**
			 * Finds all nodes by the given XPath.
			 */
			std::vector<pugi::xml_node> findNodesByXPath(const std::string &path);

			/**
			 * @get Gets an iterator over all roads.
			 */
			pugi::xml_object_range<pugi::xml_named_node_iterator> getRoads();

			/**
			 * @get Gets the header definition.
			 */
			pugi::xml_node getHeader();

			/**
			 * @get Gets an attribute of the header.
			 */
			std::string getHeader(const std::string &attribute);

			/**
			 * @get Gets the geo reference string.
			 */
			std::string getGeoReference();

		};
	}
}

#endif //CAMERASTABILIZATION_HDMAP_HPP
