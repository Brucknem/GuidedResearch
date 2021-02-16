#include "gtest/gtest.h"
#include <iostream>
#include <utility>

#include "HDMap.hpp"

using namespace providentia::calibration;

namespace providentia {
	namespace tests {

		/**
		 * Common setup for the camera tests.
		 */
		class HDMapTests : public ::testing::Test {
		protected:

			std::shared_ptr<providentia::calibration::HDMap> hdMap;

			/**
			 * @destructor
			 */
			~HDMapTests() override = default;
		};

		/**
		 * Tests that parsing does work.
		 */
		TEST_F(HDMapTests, testParsingBooks) {
			hdMap = std::make_shared<providentia::calibration::HDMap>(
				"../misc/test_xml.xodr");

			auto node = hdMap->findNode("cooking");
			std::cout << node << std::endl;
		}

	}// namespace tests
}// namespace providentia