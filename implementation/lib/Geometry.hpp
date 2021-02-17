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

		class ParamPoly3 {
		public:
			double aU;
			double bU;
			double cU;
			double dU;

			double aV;
			double bV;
			double cV;
			double dV;

			std::string pRange;

			explicit ParamPoly3(pugi::xpath_node paramPoly3Node);

			friend std::ostream &operator<<(std::ostream &os, const ParamPoly3 &obj);
		};

		class Geometry {
		public:
			double s;
			double x;
			double y;
			double hdg;
			double length;

			ParamPoly3 paramPoly3;

			PJ *projection;

			PJ_COORD latLong;

			explicit Geometry(pugi::xpath_node geometryNode, PJ *projection);

			friend std::ostream &operator<<(std::ostream &os, const Geometry &obj);

			double getLat() const {
				return latLong.lp.phi;
			}

			double getLong() const {
				return latLong.lp.lam;
			}

			std::string getLatLong() const {
				return std::to_string(getLat()) + std::string(", ") + std::to_string(getLong());
			}
		};
	}
}

#endif //CAMERASTABILIZATION_GEOMETRY_HPP
