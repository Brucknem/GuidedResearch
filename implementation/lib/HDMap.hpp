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
#include "OpenDRIVE.hpp"
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
			PJ *projection{};

			/**
			 * The OpenDRIVE roads.
			 */
			std::vector<Road> roads;

			static std::string getRoadSelector(pugi::xpath_node road);

			static std::string getRoadSelector(std::string id);

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
			 * @get
			 */
			const std::vector<Road> &getRoads() const;

			/**
			 * Finds all nodes by the given type.
			 */
			pugi::xpath_node_set findNodesByType(const std::string &type);

			/**
			 * Finds all nodes by the given XPath.
			 */
			pugi::xpath_node_set findNodesByXPath(const std::string &path);

			/**
			 * @get Gets the header definition.
			 */
			pugi::xpath_node getHeader();

			/**
			 * @get Gets an attribute of the header.
			 */
			std::string getHeader(const std::string &attribute);

			/**
			 * @get Gets the objects of the given road.
			 */
			pugi::xpath_node_set getObjects(pugi::xpath_node road);

			/**
			 * @get Gets the signals of the given road.
			 */
			pugi::xpath_node_set getSignals(pugi::xpath_node road);

			/**
			 * @get Gets the geometries of the given road.
			 */
			std::vector<OpenDRIVE> getGeometries(pugi::xpath_node road);

			/**
			 * @get Checks if the road with the given id exists.
			 */
			bool hasRoad(const std::string &id);

			/**
			 * @get Gets a specific road with the given id.
			 * @throws invalid_argument if no road with the given id is found.
			 */
			Road getRoad(const std::string &id) const;

			/**
			 * @get Gets the geo reference string.
			 */
			const std::string &getProjectionString() const;

			void parse();

			void parseProjection();
		};
	}
}

#endif //CAMERASTABILIZATION_HDMAP_HPP
