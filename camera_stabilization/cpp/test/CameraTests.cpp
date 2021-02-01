

#include "gtest/gtest.h"
#include <iostream>

#include "Camera.hpp"
#include "Intrinsics.hpp"
#include "CameraTestBase.hpp"

namespace providentia {
	namespace tests {

		/**
		 * Asserts that the camera rotation matrix is built up by the given right, up and forward vectors.
		 */
		void assertRotation(const providentia::camera::Camera &camera, const Eigen::Vector3f &expectedRight,
							const Eigen::Vector3f &expectedUp, const Eigen::Vector3f &expectedForward) {
			assertVectorsNearEqual(camera.getRotation().block<3, 1>(0, 0), expectedRight);
			assertVectorsNearEqual(camera.getRotation().block<3, 1>(0, 1), expectedUp);
			assertVectorsNearEqual(camera.getRotation().block<3, 1>(0, 2), expectedForward);
		}

		/**
		 * Asserts that the camera translation is the given expected translation.
		 */
		void assertTranslation(const providentia::camera::Camera &camera, const Eigen::Vector3f &expectedTranslation) {
			assertVectorsNearEqual(camera.getTranslation().block<3, 1>(0, 3), expectedTranslation);
		}

		class CameraTests : public CameraTestBase {
		public:
			Eigen::Vector2f imageSize{1920, 1200};
			providentia::camera::Camera camera{32, 1920.f / 1200, 20, imageSize, translation, rotation};

			~CameraTests() override = default;
		};

		/**
		 * Tests the camera rotation matrix that is built from the euler angles rotation vector.
		 */
		TEST_F(CameraTests, testCameraRotationMatrix) {
			camera.setRotation(0, 0, 0);
			assertRotation(camera, Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY(), -Eigen::Vector3f::UnitZ());

			camera.setRotation(90, 0, 0);
			assertRotation(camera, Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitZ(), Eigen::Vector3f::UnitY());

			camera.setRotation(0, 90, 0);
			assertRotation(camera, -Eigen::Vector3f::UnitZ(), Eigen::Vector3f::UnitY(), -Eigen::Vector3f::UnitX());

			camera.setRotation(0, 0, 90);
			assertRotation(camera, Eigen::Vector3f::UnitY(), -Eigen::Vector3f::UnitX(), -Eigen::Vector3f::UnitZ());

			camera.setRotation(-90, 0, 0);
			assertRotation(camera, Eigen::Vector3f::UnitX(), -Eigen::Vector3f::UnitZ(), -Eigen::Vector3f::UnitY());

			camera.setRotation(90, 90, 0);
			assertRotation(camera, -Eigen::Vector3f::UnitZ(), Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY());

			camera.setRotation(90, 0, 90);
			assertRotation(camera, Eigen::Vector3f::UnitY(), Eigen::Vector3f::UnitZ(), -Eigen::Vector3f::UnitX());

			camera.setRotation(0, 90, 90);
			assertRotation(camera, -Eigen::Vector3f::UnitZ(), -Eigen::Vector3f::UnitX(), -Eigen::Vector3f::UnitY());

			camera.setRotation(90, 90, 90);
			assertRotation(camera, -Eigen::Vector3f::UnitZ(), Eigen::Vector3f::UnitY(), -Eigen::Vector3f::UnitX());

			camera.setRotation(180, 0, 0);
			assertRotation(camera, Eigen::Vector3f::UnitX(), -Eigen::Vector3f::UnitY(), Eigen::Vector3f::UnitZ());

			camera.setRotation(0, 180, 0);
			assertRotation(camera, -Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY(), Eigen::Vector3f::UnitZ());

			camera.setRotation(0, 0, 180);
			assertRotation(camera, -Eigen::Vector3f::UnitX(), -Eigen::Vector3f::UnitY(), -Eigen::Vector3f::UnitZ());

			camera.setRotation(180, 180, 180);
			assertRotation(camera, Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY(), -Eigen::Vector3f::UnitZ());

			camera.setRotation(360, 360, 360);
			assertRotation(camera, Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY(), -Eigen::Vector3f::UnitZ());
		}

		/**
		 * Tests the camera translation matrix that is built from the translation vector.
		 */
		TEST_F(CameraTests, testCameraTranslationMatrix) {
			assertTranslation(camera, translation);

			translation = {10, 10, 10};
			camera.setTranslation(translation);
			assertTranslation(camera, translation);

			translation = {-M_PI, M_E, M_PI * M_E};
			camera.setTranslation(translation);
			assertTranslation(camera, translation);
		}

		/**
		 * Tests the camera translation matrix that is built from the translation vector.
		 */
		TEST_F(CameraTests, testCameraViewMatrix) {
			Eigen::Matrix4f expectedView;
			expectedView << 1, 0, 0, 0,
					0, 0, 1, -10,
					0, 1, 0, 5,
					0, 0, 0, 1;

			for (int row = 0; row < camera.getWorldToCameraTransformation().rows(); row++) {
				assertVectorsNearEqual(camera.getWorldToCameraTransformation().row(row),
									   expectedView.inverse().row(row));
			}
		}

		/**
		 * Tests the camera translation matrix that is built from the translation vector.
		 */
		TEST_F(CameraTests, testWorldToCameraTransformation) {
			Eigen::Vector4f pointInWorldSpace;
			Eigen::Vector4f pointInCameraSpace;

			pointInWorldSpace << 0, 10, 0, 1;
			pointInCameraSpace = camera.getWorldToCameraTransformation() * pointInWorldSpace;
			assertVectorsNearEqual(pointInCameraSpace, 0, -5, 20);

			pointInWorldSpace << 10, 0, 0, 1;
			pointInCameraSpace = camera.getWorldToCameraTransformation() * pointInWorldSpace;
			assertVectorsNearEqual(pointInCameraSpace, 10, -5, 10);

			pointInWorldSpace << 0, 0, 10, 1;
			pointInCameraSpace = camera.getWorldToCameraTransformation() * pointInWorldSpace;
			assertVectorsNearEqual(pointInCameraSpace, 0, 5, 10);

			pointInWorldSpace << -10, -10, -10, 1;
			pointInCameraSpace = camera.getWorldToCameraTransformation() * pointInWorldSpace;
			assertVectorsNearEqual(pointInCameraSpace, -10, -15, 0);
		}


		/**
		 * Tests the camera translation matrix that is built from the translation vector.
		 */
		TEST_F(CameraTests, testRenderToImage) {
			Eigen::Vector4f pointInWorldSpace;
			Eigen::Vector2f pointInCameraSpace;

			float maxDifference = 1e-3;

			pointInWorldSpace << 0, 0, 0, 1;
			pointInCameraSpace = camera * pointInWorldSpace;
			assertVectorsNearEqual(pointInCameraSpace, 960, 0);

			pointInWorldSpace << 8, 0, 0, 1;
			pointInCameraSpace = camera * pointInWorldSpace;
			assertVectorsNearEqual(pointInCameraSpace, 1920, 0);

			pointInWorldSpace << -8, 0, 0, 1;
			pointInCameraSpace = camera * pointInWorldSpace;
			assertVectorsNearEqual(pointInCameraSpace, 0, 0, maxDifference);

			pointInWorldSpace << 0, 0, 10, 1;
			pointInCameraSpace = camera * pointInWorldSpace;
			assertVectorsNearEqual(pointInCameraSpace, 960, 1200, maxDifference);

			pointInWorldSpace << 8, 0, 10, 1;
			pointInCameraSpace = camera * pointInWorldSpace;
			assertVectorsNearEqual(pointInCameraSpace, 1920, 1200, maxDifference);

			pointInWorldSpace << -8, 0, 10, 1;
			pointInCameraSpace = camera * pointInWorldSpace;
			assertVectorsNearEqual(pointInCameraSpace, 0, 1200, maxDifference);

			pointInWorldSpace << 0, 0, 5, 1;
			pointInCameraSpace = camera * pointInWorldSpace;
			assertVectorsNearEqual(pointInCameraSpace, 1920.f / 2, 1200.f / 2, maxDifference);

			for (int i = 0; i < 1000; i += 10) {
				pointInWorldSpace << 0, i, 5, 1;
				pointInCameraSpace = camera * pointInWorldSpace;
				assertVectorsNearEqual(pointInCameraSpace, 1920.f / 2, 1200.f / 2, maxDifference);
			}
		}
	}// namespace test
}// namespace providentia