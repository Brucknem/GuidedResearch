

#include "gtest/gtest.h"
#include <iostream>

#include "CameraTestBase.hpp"
#include "Intrinsics.hpp"
#include "RenderingPipeline.hpp"

namespace providentia {
	namespace tests {

		/**
		 * Test setup for the perspective projection.
		 */
		class PerspectiveProjectionTests : public ::testing::Test {
		protected:
			Eigen::Vector2d frustumParameters{1, 1000};
			Eigen::Vector3d intrinsics{8, 1920. / 1200., 4};

			/**
			 * @destructor
			 */
			~PerspectiveProjectionTests() override = default;
		};

		/**
		 * Tests the camera frustum matrix.
		 */
		TEST_F(PerspectiveProjectionTests, testCameraFrustum) {
			Eigen::Vector4d pointInFrustum, pointInCameraSpace;
			pointInCameraSpace << 0, 0, frustumParameters.x(), 1;
			pointInFrustum = providentia::camera::toFrustum<double>(frustumParameters.data(),
																	pointInCameraSpace.data());
			assertVectorsNearEqual(pointInFrustum, 0, 0, frustumParameters.x());

			pointInCameraSpace << 4, 7, frustumParameters.x(), 1;
			pointInFrustum = providentia::camera::toFrustum<double>(frustumParameters.data(),
																	pointInCameraSpace.data());
			assertVectorsNearEqual(pointInFrustum, 4, 7, frustumParameters.x());

			pointInCameraSpace << 0, 0, frustumParameters.y(), 1;
			pointInFrustum = providentia::camera::toFrustum<double>(frustumParameters.data(),
																	pointInCameraSpace.data());
			assertVectorsNearEqual(pointInFrustum, 0, 0, frustumParameters.y());
		}


		/**
		 * Tests the camera intrinsics matrix.
		 */
		TEST_F(PerspectiveProjectionTests, testCameraToClipSpace) {
			Eigen::Vector4d pointInCameraSpace, pointInClipSpace;
			pointInCameraSpace << 0, 0, frustumParameters.x(), 1;
			pointInClipSpace = providentia::camera::toClipSpace<double>(frustumParameters.data(),
																		intrinsics.data(),
																		pointInCameraSpace.data());
			assertVectorsNearEqual(pointInClipSpace, 0, 0, -1);

			pointInCameraSpace << -1, 1 / intrinsics[1], frustumParameters.x(), 1;
			pointInClipSpace = providentia::camera::toClipSpace<double>(frustumParameters.data(),
																		intrinsics.data(),

																		pointInCameraSpace.data());
			assertVectorsNearEqual(pointInClipSpace, -1, 1, -1);

			pointInCameraSpace << 1, -1 / intrinsics[1], frustumParameters.x(), 1;
			pointInClipSpace = providentia::camera::toClipSpace<double>(frustumParameters.data(),
																		intrinsics.data(),

																		pointInCameraSpace.data());
			assertVectorsNearEqual(pointInClipSpace, 1, -1, -1);

			pointInCameraSpace << 0, 0, frustumParameters.x(), 1;
			pointInCameraSpace *= frustumParameters[1];
			pointInCameraSpace.w() = 1;
			pointInClipSpace = providentia::camera::toClipSpace<double>(frustumParameters.data(),
																		intrinsics.data(),

																		pointInCameraSpace.data());
			assertVectorsNearEqual(pointInClipSpace, 0, 0, 1);

			pointInCameraSpace << 1, 1 / intrinsics[1], frustumParameters.x(), 1;
			pointInCameraSpace *= frustumParameters[1];
			pointInCameraSpace.w() = 1;
			pointInClipSpace = providentia::camera::toClipSpace<double>(frustumParameters.data(),
																		intrinsics.data(),

																		pointInCameraSpace.data());
			assertVectorsNearEqual(pointInClipSpace, 1, 1, 1);

			pointInCameraSpace << -1, -1 / intrinsics[1], frustumParameters.x(), 1;
			pointInCameraSpace *= frustumParameters[1];
			pointInCameraSpace.w() = 1;
			pointInClipSpace = providentia::camera::toClipSpace<double>(frustumParameters.data(),
																		intrinsics.data(),

																		pointInCameraSpace.data());
			assertVectorsNearEqual(pointInClipSpace, -1, -1, 1);
		}

		/**
		 * Tests the camera intrinsics matrix.
		 */
		TEST_F(PerspectiveProjectionTests, testCameraToNormalizedDeviceCoordinates) {
			Eigen::Vector4d pointInCameraSpace, pointInClipSpace;
			Eigen::Vector2d normalizedDeviceCoordinate;
			pointInCameraSpace << 0, 0, frustumParameters.x(), 1;
			pointInClipSpace = providentia::camera::toClipSpace<double>(frustumParameters.data(),
																		intrinsics.data(),

																		pointInCameraSpace.data());
			normalizedDeviceCoordinate = providentia::camera::toNormalizedDeviceCoordinates<double>(
				pointInClipSpace.data());
			assertVectorsNearEqual(normalizedDeviceCoordinate, 0, 0);

			pointInCameraSpace << -1, 1 / intrinsics[1], frustumParameters.x(), 1;
			pointInClipSpace = providentia::camera::toClipSpace<double>(frustumParameters.data(),
																		intrinsics.data(),

																		pointInCameraSpace.data());
			normalizedDeviceCoordinate = providentia::camera::toNormalizedDeviceCoordinates<double>(
				pointInClipSpace.data());
			assertVectorsNearEqual(normalizedDeviceCoordinate, -1, 1);

			pointInCameraSpace << 1, -1 / intrinsics[1], frustumParameters.x(), 1;
			pointInClipSpace = providentia::camera::toClipSpace<double>(frustumParameters.data(),
																		intrinsics.data(),

																		pointInCameraSpace.data());
			normalizedDeviceCoordinate = providentia::camera::toNormalizedDeviceCoordinates<double>(
				pointInClipSpace.data());
			assertVectorsNearEqual(normalizedDeviceCoordinate, 1, -1);

			pointInCameraSpace << 0, 0, frustumParameters.x(), 1;
			pointInCameraSpace *= frustumParameters[1];
			pointInCameraSpace.w() = 1;
			pointInClipSpace = providentia::camera::toClipSpace<double>(frustumParameters.data(),
																		intrinsics.data(),

																		pointInCameraSpace.data());
			normalizedDeviceCoordinate = providentia::camera::toNormalizedDeviceCoordinates<double>(
				pointInClipSpace.data());
			assertVectorsNearEqual(normalizedDeviceCoordinate, 0, 0);

			pointInCameraSpace << 1, 1 / intrinsics[1], frustumParameters.x(), 1;
			pointInCameraSpace *= frustumParameters[1];
			pointInCameraSpace.w() = 1;
			pointInClipSpace = providentia::camera::toClipSpace<double>(frustumParameters.data(),
																		intrinsics.data(),

																		pointInCameraSpace.data());
			normalizedDeviceCoordinate = providentia::camera::toNormalizedDeviceCoordinates<double>(
				pointInClipSpace.data());
			assertVectorsNearEqual(normalizedDeviceCoordinate, 1, 1);

			pointInCameraSpace << -1, -1 / intrinsics[1], frustumParameters.x(), 1;
			pointInCameraSpace *= frustumParameters[1];
			pointInCameraSpace.w() = 1;
			pointInClipSpace = providentia::camera::toClipSpace<double>(frustumParameters.data(),
																		intrinsics.data(),

																		pointInCameraSpace.data());
			normalizedDeviceCoordinate = providentia::camera::toNormalizedDeviceCoordinates<double>(
				pointInClipSpace.data());
			assertVectorsNearEqual(normalizedDeviceCoordinate, -1, -1);
		}
	}// namespace toCameraSpace
}// namespace providentia