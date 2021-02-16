//
// Created by brucknem on 16.02.21.
//

#ifndef CAMERASTABILIZATION_HDMAP_HPP
#define CAMERASTABILIZATION_HDMAP_HPP

#include <string>
#include <proj.h>
#include <exception>
#include <stdexcept>
#include <iostream>

#include "tinyxml2.h"

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
			tinyxml2::XMLDocument doc;

		public:

			/**
			 * @constructor
			 */
			explicit HDMap(std::string filename);

			tinyxml2::XMLElement *findNode(long id) {
				return findNode(std::to_string(id));
			}

			tinyxml2::XMLElement *findNode(const std::string &id) {
				tinyxml2::XMLElement *elem = doc.FirstChildElement(); //Tree root

				while (elem) {
					std::cout << "Yeeeeet" << std::endl;
					std::cout << elem->GetText() << std::endl;
					std::cout << elem->GetLineNum() << std::endl;
					if (std::string(elem->Value()) == id) {
						return elem;
					}
					if (elem->FirstChildElement()) {
						elem = elem->FirstChildElement();
					} else if (elem->NextSiblingElement()) {
						elem = elem->NextSiblingElement();
					} else {
						if (elem->Parent()->ToElement()->NextSiblingElement()) {
							elem = elem->Parent()->ToElement()->NextSiblingElement();
						} else {
							tinyxml2::XMLElement *childElement = elem->Parent()->ToElement()->FirstChildElement();
							if (childElement && std::string(elem->Name()) == std::string(childElement->Name())) {
								elem = childElement;
							} else {
								break;
							}
						}
					}
				}
				return nullptr;
			}
		};
	}
}

#endif //CAMERASTABILIZATION_HDMAP_HPP
