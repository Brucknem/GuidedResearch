//
// Created by brucknem on 17.02.21.
//

#ifndef CAMERASTABILIZATION_OPENDRIVE_HPP
#define CAMERASTABILIZATION_OPENDRIVE_HPP

#include "pugixml.hpp"
#include "boost/lexical_cast.hpp"
#include "proj.h"
#include <memory>

namespace providentia {
	namespace calibration {

		/**
		 * Wrapper for the OpenDRIVE header.
		 * https://www.asam.net/index.php?eID=dumpFile&t=f&f=3495&token=56b15ffd9dfe23ad8f759523c806fc1f1a90a0e8#_georeferencing_in_opendrive
		 */
		class GeoReference {

			/**
			 * The raw geo reference string.
			 */
			std::string geoReference;

			/**
			 * The geo projection string.
			 */
			std::string projection;
		};

		/**
		 * Wrapper for the OpenDRIVE header.
		 * https://www.asam.net/index.php?eID=dumpFile&t=f&f=3495&token=56b15ffd9dfe23ad8f759523c806fc1f1a90a0e8#_file_structure
		 */
		class Header {

			/**
			 * Major revision number of OpenDRIVE format
			 */
			ushort revMajor;

			/**
			 * Minor revision number of OpenDRIVE format; 6 for OpenDrive 1.6
			 */
			ushort revMinor;

			/**
			 * Database name
			 */
			std::string name;

			/**
			 * Version of this road network
			 */
			std::string version;

			/**
			 * Time/date of database creation according to ISO 8601 (preference: YYYY-MM-DDThh:mm:ss)
			 */
			std::string date;

			/**
			 * Maximum inertial y value
			 */
			double north;

			/**
			 * Minimum inertial y value
			 */
			double south;

			/**
			 * Maximum inertial x value
			 */
			double east;

			/**
			 * Minimum inertial x value
			 */
			double west;

			/**
			 * Vendor name
			 */
			std::string vendor;
		};

		/**
		 * Wrapper for the OpenDRIVE traffic rule of the road.
		 */
		enum TrafficRule {
			RHT = 0,
			LHT
		};

		/**
		 * Wrapper for the OpenDRIVE type of the road.
		 * https://www.asam.net/index.php?eID=dumpFile&t=f&f=3495&token=56b15ffd9dfe23ad8f759523c806fc1f1a90a0e8#_methods_of_elevation
		 */
		class Type {
		public:
			/**
			 * s-coordinate of start position.
			 */
			double s;

			/**
			 * The type.
			 */
			std::string type;

			/**
			 * @constructor
			 */
			explicit Type(pugi::xpath_node typeNode);

			/**
			 * @constructor
			 */
			Type(double s, std::string type);

			/**
			 * @destructor
			 */
			virtual ~Type() = default;

			/**
			 * @operator
			 */
			friend std::ostream &operator<<(std::ostream &os, const Type &obj);
		};

		/**
		 * Wrapper for the OpenDRIVE (super-)elevation of the road.
		 * https://www.asam.net/index.php?eID=dumpFile&t=f&f=3495&token=56b15ffd9dfe23ad8f759523c806fc1f1a90a0e8#_methods_of_elevation
		 */
		class Elevation {
		public:

			/**
			 * s-coordinate of start position.
			 */
			double s;

			/**
			 * Elevation polynom parameters.
			 */
			double a, b, c, d;

			/**
			 * @constructor
			 */
			explicit Elevation(pugi::xpath_node elevationNode);

			/**
			 * @destructor
			 */
			virtual ~Elevation() = default;

			/**
			 * @operator
			 */
			friend std::ostream &operator<<(std::ostream &os, const Elevation &obj);

			/**
			 * @constructor
			 */
			Elevation(double s, double a, double b, double c, double d);
		};

		/**
		 * Wrapper for the OpenDRIVE shape of the road.
		 * https://www.asam.net/index.php?eID=dumpFile&t=f&f=3495&token=56b15ffd9dfe23ad8f759523c806fc1f1a90a0e8#_methods_of_elevation
		 */
		class Shape : public Elevation {
		public:

			/**
			 * t-coordinate of start position.
			 */
			double t;

			/**
			 * @constructor
			 */
			explicit Shape(pugi::xpath_node shapeNode);

			/**
			 * @constructor
			 */
			Shape(double s, double t, double a, double b, double c, double d);

			/**
			 * @destructor
			 */
			~Shape() override = default;

			/**
			 * @operator
			 */
			friend std::ostream &operator<<(std::ostream &os, const Shape &obj);
		};

		/**
		 * Wrapper for the OpenDRIVE type paramPoly3.
		 * https://www.asam.net/index.php?eID=dumpFile&t=f&f=3495&token=56b15ffd9dfe23ad8f759523c806fc1f1a90a0e8#_parametric_cubic_curve
		 */
		class ParamPoly3 {
		public:

			/**
			 * Polynom parameters.
			 */
			double aU, bU, cU, dU;

			/**
			 * Polynom parameters.
			 */
			double aV, bV, cV, dV;

			/**
			 * Range of parameter p.
			 */
			std::string pRange;

			/**
			 * @constructor
			 */
			explicit ParamPoly3(pugi::xpath_node paramPoly3Node);

			/**
			 * @constructor
			 */
			ParamPoly3(double aU, double bU, double cU, double dU, double aV, double bV, double cV, double dV,
					   std::string pRange);

			/**
			 * @destructor
			 */
			virtual ~ParamPoly3() = default;

			/**
			 * @operator
			 */
			friend std::ostream &operator<<(std::ostream &os, const ParamPoly3 &obj);
		};

		/**
		 * Wrapper for the OpenDRIVE type geometry.
		 * https://www.asam.net/index.php?eID=dumpFile&t=f&f=3495&token=56b15ffd9dfe23ad8f759523c806fc1f1a90a0e8#_geometry
		 */
		class OpenDRIVE {
		private:
			/**
			 * A projection used to transform the x, y coordinate from the original transverse mercator projection to
			 * latitude and longitude.
			 */
			PJ *projection;

		public:

			/**
			 * s-coordinate of start position.
			 */
			double s;

			/**
			 * Start position (x inertial).
			 */
			double x;

			/**
			 * Start position (y inertial).
			 */
			double y;

			/**
			 * Start orientation (inertial heading).
			 */
			double hdg;

			/**
			 * Length of the element’s reference line.
			 */
			double length;

			/**
			 * The parametric polynomial curve defining the geometry.
			 */
			ParamPoly3 paramPoly3;

			/**
			 * The transformed coordinate.
			 */
			PJ_COORD latLong{};

			/**
			 * @constructor
			 */
			explicit OpenDRIVE(pugi::xpath_node geometryNode, PJ *projection);

			/**
			 * @constructor
			 */
			OpenDRIVE(double s, double x, double y, double hdg, double length, const ParamPoly3 &paramPoly3,
					  PJ *projection);

			/**
			 * @destructor
			 */
			virtual ~OpenDRIVE() = default;

			/**
			 * @operator
			 */
			friend std::ostream &operator<<(std::ostream &os, const OpenDRIVE &obj);

			/**
			 * @get x in latitude.
			 */
			double getLat() const;

			/**
			 * @get y in longitude.
			 */
			double getLong() const;

			/**
			 * @get A string representation of the latitude and longitude.
			 */
			std::string getLatLong() const;
		};

		/**
		 * Wrapper for the OpenDRIVE road.
		 * https://www.asam.net/index.php?eID=dumpFile&t=f&f=3495&token=56b15ffd9dfe23ad8f759523c806fc1f1a90a0e8#_roads
		 */
		class Road {
		public:
			/**
			 * Name of the road. May be chosen freely.
			 */
			std::string name;

			/**
			 * Total length of the reference line in the xy-plane. Change in length due to elevation is not considered.
			 */
			double length;

			/**
			 * Unique ID within the database. If it represents an integer number, it should comply to uint32_t and stay within the given range.
			 */
			std::string id;

			/**
			 * ID of the junction to which the road belongs as a connecting road (= -1 for none).
			 */
			std::string junction = "-1";

			/**
			 * Basic rule for using the road; RHT=right-hand traffic, LHT=left-hand traffic. When this attribute is missing, RHT is assumed.
			 */
			TrafficRule rule = TrafficRule::RHT;

			/**
			 * The type.
			 */
			std::shared_ptr<Type> type;

			/**
			 * @constructor
			 */
			explicit Road(pugi::xpath_node typeNode);

			/**
			 * @constructor
			 */
			Road(std::string name, double length, std::string id, std::string junction = "-1",
				 TrafficRule rule = TrafficRule::RHT);

			/**
			 * @destructor
			 */
			virtual ~Road() = default;

			/**
			 * @operator
			 */
			friend std::ostream &operator<<(std::ostream &os, const Road &obj);

			void setType(pugi::xpath_node typeNode);
		};
	}
}

#endif //CAMERASTABILIZATION_OPENDRIVE_HPP
