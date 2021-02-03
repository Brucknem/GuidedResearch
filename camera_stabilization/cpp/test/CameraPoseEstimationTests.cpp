#include "gtest/gtest.h"
#include <iostream>

#include "Intrinsics.hpp"
#include "CameraTestBase.hpp"
#include "CameraPoseEstimation.hpp"

namespace providentia {
	namespace tests {

		/**
		 * Common toCameraSpace setup for the camera tests.
		 */
		class CameraPoseEstimationTests : public ::testing::Test {
		protected:
			std::shared_ptr<providentia::calibration::CameraPoseEstimator> estimator;

			void SetUp() override;

			/**
			 * @destructor
			 */
			~CameraPoseEstimationTests() override = default;

		};

		void CameraPoseEstimationTests::SetUp() {
			estimator = std::make_shared<providentia::calibration::CameraPoseEstimator>(
					Eigen::Vector3d{0, -10, 5}, Eigen::Vector3d{90, 0, 0}
			);
		}

		/**
		 * Tests the camera rotation matrix that is built from the euler angles rotation vector.
		 */
		TEST_F(CameraPoseEstimationTests, testEstimationFromOriginalTransformation) {
//			google::InitGoogleLogging("Test");

			Eigen::Vector4d pointInWorldSpace;

			pointInWorldSpace << 0, 10, 0, 1;
			estimator->addReprojectionResidual(pointInWorldSpace, {0., 0.});
//
//			pointInWorldSpace << 0, 10, 5, 1;
//			estimator->addReprojectionResidual(pointInWorldSpace, *camera * pointInWorldSpace);
//
//			pointInWorldSpace << -4, 15, 3, 1;
//			estimator->addReprojectionResidual(pointInWorldSpace, *camera * pointInWorldSpace);

			estimator->solve();
		}
	}// namespace toCameraSpace
}// namespace providentia