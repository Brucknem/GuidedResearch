

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

		/**
		 * Common test setup for the camera tests.
		 */
		class CameraTests : public CameraTestBase {
		protected:
			/**
			 * A test image size.
			 */
			Eigen::Vector2f imageSize{1920, 1200};

			/**
			 * A test camera.
			 */
			providentia::camera::Camera camera{32, 1920.f / 1200, 20, imageSize, translation, rotation};

			/**
			 * A default maximal difference for vector elements.
			 */
			float maxDifference = 1e-3;

			/**
			 * @destructor
			 */
			~CameraTests() override = default;

			/**
			 *	Asserts that a point on the ray shot from the camera is rendered to the expected pixel.
			 *
			 * @param direction The direction of the ray from the camera center in camera space.
			 * @param expectedX The expected x pixel location.
			 * @param expectedY The expected y pixel location.
			 * @param _maxDifference	The maximal difference in pixel space.
			 */
			void
			assertPointInCameraSpaceOnPixel(const Eigen::Vector3f &direction,
											const float expectedX, float expectedY,
											float _maxDifference = 1e-4) {
				for (int i = 1; i < 20; ++i) {
					pointInCameraSpace.head<3>() = direction * i;
					pointInCameraSpace.w() = 1;
					pointInWorldSpace = camera.getCameraToWorldTransformation() * pointInCameraSpace;
					pointInImageSpace = camera * pointInWorldSpace;
					assertVectorsNearEqual(pointInImageSpace, Eigen::Vector2f(expectedX, expectedY), _maxDifference);
				}
			}

			/**
			 * Asserts that some sampled points along the ray through a corner of the image plane
			 * is rendered onto the corner after the projection.
			 *
			 * @param imagePlaneCorner The corner of the image plane.
			 * @param expectedX The expected x pixel location.
			 * @param expectedY The expected y pixel location.
			 * @param _maxDifference	The maximal difference in pixel space.
			 */
			void assertPointOnCorner(const Eigen::Vector2f &imagePlaneCorner,
									 const float expectedX, float expectedY,
									 float _maxDifference = 1e-3) {
				assertPointInCameraSpaceOnPixel(Eigen::Vector3f{imagePlaneCorner.x(), imagePlaneCorner.y(),
																(float) camera.getPerspectiveProjection()->getFrustumPlaneDistances().x()},
												expectedX, expectedY, _maxDifference);
			}
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
		 * Tests the camera view matrix that is built from the translation and rotation vector.
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
		 * Tests the world to camera transformation.
		 */
		TEST_F(CameraTests, testWorldToCameraTransformation) {
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
		 * Tests that points on the ray along the corners of the image plane all fall to the respective corner pixel.
		 */
		TEST_F(CameraTests, testCornerPixels) {
			Eigen::Vector2f direction = camera.getPerspectiveProjection()->getLowerLeft();
			assertPointOnCorner(direction, 0, 0);

			direction.y() *= -1;
			assertPointOnCorner(direction, 0, imageSize.y());

			direction = camera.getPerspectiveProjection()->getTopRight();
			assertPointOnCorner(direction, imageSize.x(), imageSize.y());

			direction.y() *= -1;
			assertPointOnCorner(direction, imageSize.x(), 0);
		}

		/**
		 * Tests that rendering points in world coordinates results in correct points in pixel coordinates.
		 */
		TEST_F(CameraTests, testRenderToImage) {
			pointInWorldSpace << 0, 0, 0, 1;
			pointInImageSpace = camera * pointInWorldSpace;
			assertVectorsNearEqual(pointInImageSpace, 960, 0);

			pointInWorldSpace << 8, 0, 0, 1;
			pointInImageSpace = camera * pointInWorldSpace;
			assertVectorsNearEqual(pointInImageSpace, 1920, 0);

			pointInWorldSpace << -8, 0, 0, 1;
			pointInImageSpace = camera * pointInWorldSpace;
			assertVectorsNearEqual(pointInImageSpace, 0, 0, maxDifference);

			pointInWorldSpace << 0, 0, 10, 1;
			pointInImageSpace = camera * pointInWorldSpace;
			assertVectorsNearEqual(pointInImageSpace, 960, 1200, maxDifference);

			pointInWorldSpace << 8, 0, 10, 1;
			pointInImageSpace = camera * pointInWorldSpace;
			assertVectorsNearEqual(pointInImageSpace, 1920, 1200, maxDifference);

			pointInWorldSpace << -8, 0, 10, 1;
			pointInImageSpace = camera * pointInWorldSpace;
			assertVectorsNearEqual(pointInImageSpace, 0, 1200, maxDifference);

			for (int i = -8; i < 1000; i += 10) {
				pointInWorldSpace << 0, i, 5, 1;
				pointInImageSpace = camera * pointInWorldSpace;
				assertVectorsNearEqual(pointInImageSpace, 1920.f / 2, 1200.f / 2, maxDifference);
			}
		}
	}// namespace test
}// namespace providentia