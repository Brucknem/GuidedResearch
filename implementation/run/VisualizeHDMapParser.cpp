//
// Created by brucknem on 13.01.21.
//
#include <iostream>
#include "HDMap.hpp"

int main(int argc, char const *argv[]) {
	providentia::calibration::HDMap hdMap("../misc/test_xml.xodr");

	auto nodes = hdMap.findNodes("book");

	for (const auto &node : nodes) {
		std::cout << node.name() << std::endl;
	}

	return 0;
}
