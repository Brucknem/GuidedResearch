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
			 * @operator
			 */
			friend std::ostream &operator<<(std::ostream &os, const Geometry &obj);

			/**
			 * @get x in latitude.
			 */
			double getLat() const {
				return latLong.lp.phi;
			}

			/**
			 * @get y in longitude.
			 */
			double getLong() const {
				return latLong.lp.lam;
			}

			/**
			 * @get A string representation of the latitude and longitude.
			 */
			std::string getLatLong() const {
				return std::to_string(getLat()) + std::string(", ") + std::to_string(getLong());
			}
		};
	}
}

#endif //CAMERASTABILIZATION_GEOMETRY_HPP
