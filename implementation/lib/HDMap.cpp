//
// Created by brucknem on 16.02.21.
//

#include "HDMap.hpp"

#include <utility>

namespace providentia {
	namespace calibration {
		HDMap::HDMap(std::string _filename) : filename(std::move(_filename)) {
			doc.LoadFile(filename.c_str());

			if (doc.ErrorID() != tinyxml2::XML_SUCCESS) {
				// XML file is not ok ... we throw some exception
				throw std::invalid_argument("XML file parsing failed: " + std::to_string(doc.ErrorID()));
			} // if
		}
	}
}
