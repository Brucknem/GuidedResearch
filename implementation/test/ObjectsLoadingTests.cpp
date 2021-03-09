//
// Created by brucknem on 04.02.21.
//

#include "gtest/gtest.h"
#include "ObjectsLoading.hpp"
#include "yaml-cpp/yaml.h"

using namespace providentia::calibration;

namespace providentia {
	namespace tests {

		/**
		 * Tests for the residual blocks.
		 */
		class ObjectsLoadingTests : public ::testing::Test {
		protected:

			/**
			 * @destructor
			 */
			~ObjectsLoadingTests() override = default;
		};

		/**
		 * Tests loading the objects from a YAML file.
		 */
		TEST_F(ObjectsLoadingTests, testLoadingObjects) {
			auto objects = LoadObjects("../misc/objects.yaml", "../misc/pixels.yaml");

			ASSERT_EQ(objects.size(), 554);
		}
	}
}

