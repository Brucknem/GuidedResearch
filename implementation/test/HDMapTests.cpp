#include "gtest/gtest.h"
#include <iostream>
#include <utility>
#include <cstring>
#include <algorithm>

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
		 * Tests that parsing the real HD map works.
		 */
		TEST_F(HDMapTests, testParsingTestMap) {
			hdMap = std::make_shared<providentia::calibration::HDMap>("../misc/map_snippet.xodr");

			ASSERT_STREQ(hdMap->getHeader("north").c_str(), "5.350576134016e+06");
			ASSERT_STREQ(hdMap->getHeader("vendor").c_str(), "3D Mapping Solutions");

			ASSERT_STREQ(hdMap->getProjectionString().c_str(),
						 "+proj=tmerc +lat_0=0 +lon_0=9 +k=0.9996 +x_0=500000 +y_0=0 +datum=WGS84 +units=m +no_defs");

			for (const auto &road : hdMap->getRoads()) {
				ASSERT_STREQ(road.node().name(), "road");
			}

			auto road = hdMap->getRoads()[0].node();
			ASSERT_EQ(hdMap->getSignals(road).size(), 14);
			ASSERT_EQ(hdMap->getObjects(road).size(), 24);

			road = hdMap->getRoads()[6].node();
			ASSERT_EQ(hdMap->getSignals(road).size(), 62);
			ASSERT_EQ(hdMap->getObjects(road).size(), 72);
		}

		/**
		 * Tests the geometry counts in the map.
		 */
		TEST_F(HDMapTests, testGetGeometry) {
			hdMap = std::make_shared<providentia::calibration::HDMap>("../misc/map_snippet.xodr");

			std::vector<const char *> ss{
				"0.000000000000e+00",
				"2.874078777576e+02",
				"8.622236665343e+02",
				"1.437039521207e+03"
			};

			int geometries = 0;
			for (const auto &road : hdMap->getRoads()) {
				for (const auto &geometry : hdMap->getGeometries(road)) {
					geometries++;
				}
			}

			const char *id = "2311000";
			ASSERT_EQ(hdMap->hasRoad(id), true);

			int i = 0;
			for (const auto &geometry : hdMap->getGeometries(hdMap->getRoad(id))) {
				EXPECT_NEAR(geometry.s, boost::lexical_cast<double>(ss[i++]), 1e-8);
			}

			ASSERT_EQ(geometries, 1421);
		}

		/**
		 * Tests finding a pole.
		 */
		TEST_F(HDMapTests, testLatLong) {
			hdMap = std::make_shared<providentia::calibration::HDMap>("../misc/map_snippet.xodr");

			std::vector<double> lat{
				48.246232309984222,
				48.243781827991008,
				48.238880414687834,
				48.233978324644234
			};

			std::vector<double> lon{
				11.641820194151922,
				11.640590221947996,
				11.638132431508589,
				11.6356778613078
			};

			const char *id = "2311000";
			ASSERT_EQ(hdMap->hasRoad(id), true);

			int i = 0;
			for (const auto &geometry : hdMap->getGeometries(hdMap->getRoad(id))) {
				EXPECT_NEAR(geometry.getLat(), lat[i], 1e-8);
				EXPECT_NEAR(geometry.getLong(), lon[i], 1e-8);
				i++;
			}
		}
	}// namespace tests
}// namespace providentia