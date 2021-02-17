//
// Created by brucknem on 16.02.21.
//

#ifndef CAMERASTABILIZATION_HDMAP_HPP
#define CAMERASTABILIZATION_HDMAP_HPP

#include <string>
#include <proj.h>
#include <stdexcept>
#include <iostream>
#include <vector>

#include "pugixml.hpp"
#include "Geometry.hpp"
#include "proj.h"

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

			/**
			 * The projectionString defining the coordinate system.
			 */
			std::string projectionString;

			/**
			 * The coordinate transformation.
			 */
			PJ *projection;

		public:

			/**
			 * @constructor
			 */
			explicit HDMap(std::string filename);

			/**
			 * @destructor
			 */
			virtual ~HDMap();

			/**
			 * Finds all nodes by the given type.
			 */
			pugi::xpath_node_set findNodesByType(const std::string &type);

			/**
			 * Finds all nodes by the given XPath.
			 */
			pugi::xpath_node_set findNodesByXPath(const std::string &path);

			/**
			 * @get Gets an iterator over all roads.
			 */
			pugi::xpath_node_set getRoads();

			/**
			 * @get Gets the header definition.
			 */
			pugi::xpath_node getHeader();

			/**
			 * @get Gets an attribute of the header.
			 */
			std::string getHeader(const std::string &attribute);

			pugi::xpath_node_set getObjects(pugi::xpath_node road);

			static std::string getRoadSelector(pugi::xpath_node road);

			static std::string getRoadSelector(std::string id);

			pugi::xpath_node_set getSignals(pugi::xpath_node road);

			std::vector<Geometry> getGeometries(pugi::xpath_node road);

			pugi::xpath_node getRoad(const std::string &id);

			/**
			 * @get Gets the geo reference string.
			 */
			const std::string &getProjectionString() const;
		};
	}
}

#endif //CAMERASTABILIZATION_HDMAP_HPP
