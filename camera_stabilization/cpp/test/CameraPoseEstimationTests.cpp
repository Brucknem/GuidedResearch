#include "gtest/gtest.h"
#include <iostream>

#include "Camera.hpp"
#include "Intrinsics.hpp"
#include "CameraTestBase.hpp"
#include "CameraPoseEstimation.hpp"
#include "ceres/ceres.h"

namespace providentia {
	namespace tests {

		/**
		 * Common test setup for the camera tests.
		 */
		class CameraPoseEstimationTests : public ::testing::Test {
		protected:
			std::shared_ptr<providentia::camera::Camera> camera;
			std::shared_ptr<providentia::calibration::CameraPoseEstimator> estimator;

			void SetUp() override;

			/**
			 * @destructor
			 */
			~CameraPoseEstimationTests() override = default;

		};

		void CameraPoseEstimationTests::SetUp() {
			camera = std::make_shared<providentia::camera::BlenderCamera>();
			estimator = std::make_shared<providentia::calibration::CameraPoseEstimator>(*camera);
		}

		/**
		 * Tests the camera rotation matrix that is built from the euler angles rotation vector.
		 */
		TEST_F(CameraPoseEstimationTests, testEstimationFromOriginalTransformation) {
//			google::InitGoogleLogging("Test");

			Eigen::Vector4d pointInWorldSpace;

			pointInWorldSpace << 0, 0, 0, 1;
			estimator->addReprojectionResidual(pointInWorldSpace, *camera * pointInWorldSpace);

			pointInWorldSpace << 0, 10, 5, 1;
			estimator->addReprojectionResidual(pointInWorldSpace, *camera * pointInWorldSpace);

			pointInWorldSpace << -4, 15, 3, 1;
			estimator->addReprojectionResidual(pointInWorldSpace, *camera * pointInWorldSpace);

			estimator->solve();
		}

	}// namespace test
}// namespace providentia