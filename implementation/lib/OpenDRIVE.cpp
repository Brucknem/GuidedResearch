//
// Created by brucknem on 17.02.21.
//

#include "OpenDRIVE.hpp"

#include <utility>
#include <memory>

namespace providentia {
	namespace calibration {
#pragma region Road

		Road::Road(std::string name, double length, std::string id, std::string junction,
				   TrafficRule rule) : name(std::move(name)), length(length), id(std::move(id)),
									   junction(std::move(junction)), rule(rule) {}

		Road::Road(pugi::xpath_node typeNode) : Road(
			typeNode.node().attribute("name").value(),
			boost::lexical_cast<double>(typeNode.node().attribute("length").value()),
			typeNode.node().attribute("id").value()
		) {
			std::string _junction = typeNode.node().attribute("junction").value();
			if (!_junction.empty()) {
				junction = std::move(_junction);
			}
			std::string _rule = typeNode.node().attribute("rule").value();
			if (!_rule.empty() && _rule == "LHT") {
				rule = TrafficRule::LHT;
			}
		}

		std::ostream &operator<<(std::ostream &os, const Road &obj) {
			os << "name=\"" << obj.name << std::endl;
			os << "length=\"" << obj.length << std::endl;
			os << "id=\"" << obj.id << std::endl;
			os << "junction=\"" << obj.junction << std::endl;
			os << "rule=\"" << obj.rule << std::endl;
			return os;
		}

		void Road::setType(pugi::xpath_node typeNode) {
			type = std::make_shared<Type>(typeNode);
		}

#pragma endregion Road

#pragma region ParamPoly3

		ParamPoly3::ParamPoly3(double aU, double bU, double cU, double dU, double aV, double bV, double cV, double dV,
							   std::string pRange) : aU(aU), bU(bU), cU(cU), dU(dU), aV(aV), bV(bV), cV(cV),
													 dV(dV), pRange(std::move(pRange)) {}

		ParamPoly3::ParamPoly3(pugi::xpath_node paramPoly3Node) : ParamPoly3(
			boost::lexical_cast<double>(paramPoly3Node.node().attribute("aU").value()),
			boost::lexical_cast<double>(paramPoly3Node.node().attribute("bU").value()),
			boost::lexical_cast<double>(paramPoly3Node.node().attribute("cU").value()),
			boost::lexical_cast<double>(paramPoly3Node.node().attribute("dU").value()),
			boost::lexical_cast<double>(paramPoly3Node.node().attribute("aV").value()),
			boost::lexical_cast<double>(paramPoly3Node.node().attribute("bV").value()),
			boost::lexical_cast<double>(paramPoly3Node.node().attribute("cV").value()),
			boost::lexical_cast<double>(paramPoly3Node.node().attribute("dV").value()),
			paramPoly3Node.node().attribute("pRange").value()) {}

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

#pragma endregion ParamPoly3

#pragma region Geometry

		OpenDRIVE::OpenDRIVE(double s, double x, double y, double hdg, double length, const ParamPoly3 &paramPoly3,
							 PJ *projection) : projection(projection), s(s), x(x), y(y), hdg(hdg),
											   length(length), paramPoly3(paramPoly3) {
			latLong = proj_trans(
				projection,
				PJ_FWD,
				proj_coord(x, y, 0, 0)
			);
		}

		OpenDRIVE::OpenDRIVE(pugi::xpath_node geometryNode, PJ *_projection) : OpenDRIVE(
			boost::lexical_cast<double>(geometryNode.node().attribute("s").value()),
			boost::lexical_cast<double>(geometryNode.node().attribute("x").value()),
			boost::lexical_cast<double>(geometryNode.node().attribute("y").value()),
			boost::lexical_cast<double>(geometryNode.node().attribute("hdg").value()),
			boost::lexical_cast<double>(geometryNode.node().attribute("length").value()),
			ParamPoly3{geometryNode.node().child("paramPoly3")},
			_projection) {}

		std::ostream &operator<<(std::ostream &os, const OpenDRIVE &obj) {
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

		std::string OpenDRIVE::getLatLong() const {
			return std::to_string(getLat()) + std::string(", ") + std::to_string(getLong());
		}

		double OpenDRIVE::getLong() const {
			return latLong.lp.lam;
		}

		double OpenDRIVE::getLat() const {
			return latLong.lp.phi;
		}

#pragma endregion Geometry

#pragma region Elevation

		Elevation::Elevation(double s, double a, double b, double c, double d) : s(s), a(a), b(b), c(c), d(d) {}

		Elevation::Elevation(pugi::xpath_node paramPoly3Node) : Elevation(
			boost::lexical_cast<double>(paramPoly3Node.node().attribute("s").value()),
			boost::lexical_cast<double>(paramPoly3Node.node().attribute("a").value()),
			boost::lexical_cast<double>(paramPoly3Node.node().attribute("b").value()),
			boost::lexical_cast<double>(paramPoly3Node.node().attribute("c").value()),
			boost::lexical_cast<double>(paramPoly3Node.node().attribute("d").value())) {}

		std::ostream &operator<<(std::ostream &os, const Elevation &obj) {
			os << "s=\"" << obj.s << std::endl;
			os << "a=\"" << obj.a << std::endl;
			os << "b=\"" << obj.b << std::endl;
			os << "c=\"" << obj.c << std::endl;
			os << "d=\"" << obj.d << std::endl;
			return os;
		}

#pragma endregion Elevation

#pragma region Shape

		Shape::Shape(double s, double t, double a, double b, double c, double d) : Elevation(s, a, b, c, d), t(t) {}

		std::ostream &operator<<(std::ostream &os, const Shape &obj) {
			os << static_cast<const Elevation &>(obj) << std::endl;
			os << "t=\"" << obj.t << std::endl;
			return os;
		}

		Shape::Shape(pugi::xpath_node paramPoly3Node) :
			Elevation(paramPoly3Node),
			t(boost::lexical_cast<double>(paramPoly3Node.node().attribute("t").value())) {}

#pragma endregion Shape

#pragma region Type

		Type::Type(double s, std::string type) : s(s), type(std::move(type)) {}

		Type::Type(pugi::xpath_node typeNode) : Type(
			boost::lexical_cast<double>(typeNode.node().attribute("s").value()),
			typeNode.node().attribute("type").value()
		) {}

		std::ostream &operator<<(std::ostream &os, const Type &obj) {
			os << "s=\"" << obj.s << std::endl;
			os << "type=\"" << obj.type << std::endl;
			return os;
		}

#pragma endregion Type

	}
}