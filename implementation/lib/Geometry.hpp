//
// Created by brucknem on 17.02.21.
//

#ifndef CAMERASTABILIZATION_GEOMETRY_HPP
#define CAMERASTABILIZATION_GEOMETRY_HPP

#include "pugixml.hpp"
#include "boost/lexical_cast.hpp"
#include "proj.h"

namespace providentia {
	namespace calibration {

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
			explicit Elevation(pugi::xpath_node paramPoly3Node);

			/**
			 * @constructor
			 */
			Elevation(double s, double a, double b, double c, double d);

			/**
			 * @destructor
			 */
			virtual ~Elevation() = default;

			/**
			 * @operator
			 */
			friend std::ostream &operator<<(std::ostream &os, const Elevation &obj);
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
			explicit Shape(pugi::xpath_node paramPoly3Node);

			/**
			 * @constructor
			 */
			Shape(double s, double t, double a, double b, double c, double d);

			/**
			 * @destructor
			 */
			virtual ~Shape() = default;

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
		class Geometry {
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
			PJ_COORD latLong;

			/**
			 * @constructor
			 */
			explicit Geometry(pugi::xpath_node geometryNode, PJ *projection);

			/**
			 * @constructor
			 */
			Geometry(double s, double x, double y, double hdg, double length, const ParamPoly3 &paramPoly3,
					 PJ *projection);

			/**
			 * @destructor
			 */
			virtual ~Geometry() = default;

			/**
			 * @operator
			 */
			friend std::ostream &operator<<(std::ostream &os, const Geometry &obj);

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
	}
}

#endif //CAMERASTABILIZATION_GEOMETRY_HPP
