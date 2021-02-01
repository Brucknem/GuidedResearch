

#include "gtest/gtest.h"
#include <iostream>

#include "Camera.hpp"
#include "CameraTestBase.hpp"
#include "Intrinsics.hpp"
#include "PerspectiveProjection.hpp"

namespace providentia {
	namespace tests {

		class PerspectiveProjectionTests : public CameraTestBase {
		public:
			float aspect = 1920.f / 1200;
			providentia::camera::PerspectiveProjection perspectiveProjection{8, aspect, 4};

			~PerspectiveProjectionTests() override = default;
		};

		/**
		 * Tests the camera frustum matrix.
		 */
		TEST_F(PerspectiveProjectionTests, testCameraFrustum) {
			Eigen::Vector4f pointInFrustum;
			pointInCameraSpace << 0, 0, 1, 1;
			pointInFrustum = perspectiveProjection.toFrustum(pointInCameraSpace);
			assertVectorsNearEqual(pointInFrustum, 0, 0, 1);

			pointInCameraSpace << 4, 7, 1, 1;
			pointInFrustum = perspectiveProjection.toFrustum(pointInCameraSpace);
			assertVectorsNearEqual(pointInFrustum, 4, 7, 1);

			pointInCameraSpace << 0, 0, 1000, 1;
			pointInFrustum = perspectiveProjection.toFrustum(pointInCameraSpace);
			assertVectorsNearEqual(pointInFrustum, 0, 0, 1000);
		}


		/**
		 * Tests the camera intrinsics matrix.
		 */
		TEST_F(PerspectiveProjectionTests, testCameraToClipSpace) {
			Eigen::Vector3f pointInClipSpace;
			pointInCameraSpace << 0, 0, 1, 1;
			pointInClipSpace = perspectiveProjection.toClipSpace(pointInCameraSpace);
			assertVectorsNearEqual(pointInClipSpace, 0, 0, -1);

			pointInCameraSpace << -1, 1 / aspect, 1, 1;
			pointInClipSpace = perspectiveProjection.toClipSpace(pointInCameraSpace);
			std::cout << pointInClipSpace << std::endl;
			assertVectorsNearEqual(pointInClipSpace, -1, 1, -1);

			pointInCameraSpace << 1, -1 / aspect, 1, 1;
			pointInClipSpace = perspectiveProjection.toClipSpace(pointInCameraSpace);
			assertVectorsNearEqual(pointInClipSpace, 1, -1, -1);

			pointInCameraSpace << 0, 0, 1, 1;
			pointInCameraSpace *= 1000;
			pointInCameraSpace.w() = 1;
			pointInClipSpace = perspectiveProjection.toClipSpace(pointInCameraSpace);
			assertVectorsNearEqual(pointInClipSpace, 0, 0, 1);

			pointInCameraSpace << 1, 1 / aspect, 1, 1;
			pointInCameraSpace *= 1000;
			pointInCameraSpace.w() = 1;
			pointInClipSpace = perspectiveProjection.toClipSpace(pointInCameraSpace);
			assertVectorsNearEqual(pointInClipSpace, 1, 1, 1);

			pointInCameraSpace << -1, -1 / aspect, 1, 1;
			pointInCameraSpace *= 1000;
			pointInCameraSpace.w() = 1;
			pointInClipSpace = perspectiveProjection.toClipSpace(pointInCameraSpace);
			assertVectorsNearEqual(pointInClipSpace, -1, -1, 1);
		}

		/**
		 * Tests the camera intrinsics matrix.
		 */
		TEST_F(PerspectiveProjectionTests, testCameraToNormalizedDeviceCoordinates) {
			Eigen::Vector2f normalizedDeviceCoordinate;
			pointInCameraSpace << 0, 0, 1, 1;
			normalizedDeviceCoordinate = perspectiveProjection * pointInCameraSpace;
			assertVectorsNearEqual(normalizedDeviceCoordinate, 0, 0);

			pointInCameraSpace << -1, 1 / aspect, 1, 1;
			normalizedDeviceCoordinate = perspectiveProjection * pointInCameraSpace;
			assertVectorsNearEqual(normalizedDeviceCoordinate, -1, 1);

			pointInCameraSpace << 1, -1 / aspect, 1, 1;
			normalizedDeviceCoordinate = perspectiveProjection * pointInCameraSpace;
			assertVectorsNearEqual(normalizedDeviceCoordinate, 1, -1);

			pointInCameraSpace << 0, 0, 1, 1;
			pointInCameraSpace *= 1000;
			pointInCameraSpace.w() = 1;
			normalizedDeviceCoordinate = perspectiveProjection * pointInCameraSpace;
			assertVectorsNearEqual(normalizedDeviceCoordinate, 0, 0);

			pointInCameraSpace << 1, 1 / aspect, 1, 1;
			pointInCameraSpace *= 1000;
			pointInCameraSpace.w() = 1;
			normalizedDeviceCoordinate = perspectiveProjection * pointInCameraSpace;
			assertVectorsNearEqual(normalizedDeviceCoordinate, 1, 1);

			pointInCameraSpace << -1, -1 / aspect, 1, 1;
			pointInCameraSpace *= 1000;
			pointInCameraSpace.w() = 1;
			normalizedDeviceCoordinate = perspectiveProjection * pointInCameraSpace;
			assertVectorsNearEqual(normalizedDeviceCoordinate, -1, -1);
		}
	}// namespace test
}// namespace providentia