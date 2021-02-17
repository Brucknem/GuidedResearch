//
// Created by brucknem on 17.02.21.
//

#include "Geometry.hpp"

#include <utility>

namespace providentia {
	namespace calibration {
//
//		double cast(std::string value) {
//			std::string delimiter = "e";
//			int index = value.find(delimiter);
//			std::string base = value.substr(0, index);
//			std::string exponent = value.substr(index + 1, value.size());
//
//			double b = std::atof(base.c_str());
//			double e = std::atof(exponent.c_str());
//
//			return b * std::pow(10, e);
//		}

		ParamPoly3::ParamPoly3(pugi::xpath_node paramPoly3Node) :
			aU(boost::lexical_cast<double>(paramPoly3Node.node().attribute("aU").value())),
			bU(boost::lexical_cast<double>(paramPoly3Node.node().attribute("bU").value())),
			cU(boost::lexical_cast<double>(paramPoly3Node.node().attribute("cU").value())),
			dU(boost::lexical_cast<double>(paramPoly3Node.node().attribute("dU").value())),
			aV(boost::lexical_cast<double>(paramPoly3Node.node().attribute("aV").value())),
			bV(boost::lexical_cast<double>(paramPoly3Node.node().attribute("bV").value())),
			cV(boost::lexical_cast<double>(paramPoly3Node.node().attribute("cV").value())),
			dV(boost::lexical_cast<double>(paramPoly3Node.node().attribute("dV").value())),
			pRange(paramPoly3Node.node().attribute("pRange").value()) {}

		std::ostream &operator<<(std::ostream &os, const ParamPoly3 &obj) {
			os << "aU=\"" << obj.aU << std::endl;
			os << "bU=\"" << obj.bU << std::endl;
			os << "cU=\"" << obj.cU << std::endl;
			os << "dU=\"" << obj.dU << std::endl;

			os << "aV=\"" << obj.aV << std::endl;
			os << "bV=\"" << obj.bV << std::endl;
			os << "cV=\"" << obj.cV << std::endl;
			os << "dV=\"" << obj.dV << std::endl;

			os << "pRange=\"" << obj.pRange;
			return os;
		}

		Geometry::Geometry(pugi::xpath_node geometryNode, PJ *_projection) :
			s(boost::lexical_cast<double>(geometryNode.node().attribute("s").value())),
			x(boost::lexical_cast<double>(geometryNode.node().attribute("x").value())),
			y(boost::lexical_cast<double>(geometryNode.node().attribute("y").value())),
			hdg(boost::lexical_cast<double>(geometryNode.node().attribute("hdg").value())),
			length(boost::lexical_cast<double>(geometryNode.node().attribute("length").value())),
			paramPoly3(ParamPoly3{geometryNode.node().child("paramPoly3")}),
			projection(_projection) {
			latLong = proj_trans(
				projection,
				PJ_FWD,
				proj_coord(x, y, 0, 0)
			);
		}

		std::ostream &operator<<(std::ostream &os, const Geometry &obj) {
			os << "s=\"" << obj.s << std::endl;
			os << "x=\"" << obj.x << std::endl;
			os << "y=\"" << obj.y << std::endl;
			os << "hdg=\"" << obj.hdg << std::endl;
			os << "length=\"" << obj.length << std::endl;
			os << "latLong=[" << obj.getLatLong() << "]" << std::endl;
			os << "paramPoly3=[" << std::endl;
			os << obj.paramPoly3 << std::endl;
			os << "]";
			return os;
		}
	}
}