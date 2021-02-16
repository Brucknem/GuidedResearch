#include "gtest/gtest.h"
#include <iostream>
#include <utility>

#include "HDMap.hpp"

using namespace providentia::calibration;

namespace providentia {
	namespace tests {

		/**
		 * Common setup for the HD Map tests.
		 */
		class HDMapTests : public ::testing::Test {
		protected:

			/**
			 * The HD map.
			 */
			std::shared_ptr<providentia::calibration::HDMap> hdMap;

			/**
			 * @destructor
			 */
			~HDMapTests() override = default;
		};

		/**
		 * Tests that parsing some clearly and artificial XML data does work.
		 */
		TEST_F(HDMapTests, testParsingBooks) {
			hdMap = std::make_shared<providentia::calibration::HDMap>("../misc/test_xml.xodr");

			auto nodes = hdMap->findNodesByType("book");
			std::vector<const char *> categories{"cooking", "children", "web", "web"};
			std::vector<const char *> titles{"Everyday Italian", "Harry Potter", "XQuery Kick Start", "Learning XML"};

			for (int i = 0; i < nodes.size(); ++i) {
				auto node = nodes[i];
				ASSERT_STREQ(node.name(), "book");
				ASSERT_STREQ(node.attribute("category").as_string(), categories[i]);
				ASSERT_STREQ(node.child("title").text().as_string(), titles[i]);
			}
		}

		/**
		 * Tests that parsing the real HD map works.
		 */
		TEST_F(HDMapTests, testParsingTestMap) {
			hdMap = std::make_shared<providentia::calibration::HDMap>("../misc/map_snippet.xodr");

			for (const auto &road : hdMap->getRoads()) {
				ASSERT_STREQ(road.name(), "road");
			}

			ASSERT_STREQ(hdMap->getHeader("north").c_str(), "5.350576134016e+06");
			ASSERT_STREQ(hdMap->getHeader("vendor").c_str(), "3D Mapping Solutions");

			ASSERT_STREQ(hdMap->getGeoReference().c_str(),
						 "+proj=tmerc +lat_0=0 +lon_0=9 +k=0.9996 +x_0=500000 +y_0=0 +datum=WGS84 +units=m +no_defs");
		}
	}// namespace tests
}// namespace providentia