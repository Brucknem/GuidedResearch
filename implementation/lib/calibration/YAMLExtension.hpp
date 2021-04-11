//
// Created by brucknem on 11.04.21.
//

#ifndef CAMERASTABILIZATION_YAMLEXTENSION_HPP
#define CAMERASTABILIZATION_YAMLEXTENSION_HPP

#include <Eigen/Core>
#include "yaml-cpp/yaml.h"

namespace YAML {
	template<>
	struct convert<Eigen::Vector3d> {
		static bool decode(const Node &node, Eigen::Vector3d &rhs);
	};

	template<>
	struct convert<Eigen::Vector2d> {
		static bool decode(const Node &node, Eigen::Vector2d &rhs);
	};

}
#endif //CAMERASTABILIZATION_YAMLEXTENSION_HPP
