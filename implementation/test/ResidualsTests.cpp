//
// Created by brucknem on 04.02.21.
//

#include "gtest/gtest.h"
#include "CameraTestBase.hpp"
#include "Residuals.hpp"

#include <utility>

using namespace providentia::calibration;

namespace providentia {
	namespace tests {

		/**
		 * Tests for the residual blocks.
		 */
		class ResidualsTest : public CameraTestBase {
		protected:

			/**
			 * @destructor
			 */
			~ResidualsTest() override = default;

			void assertParametricPoint(ParametricPoint point, Eigen::Vector4d expectedResidual) {
				Eigen::Vector4d residual;
				CorrespondenceResidual correspondenceResidual = {
					point.getExpectedPixel(), std::make_shared<ParametricPoint>(point),
					intrinsics,
					5,
					5,
					1
				};
				correspondenceResidual(translation.data(), rotation.data(), point.getLambda(), point.getMu(),
									   point.getAngle(), residual.data());

				EXPECT_NEAR(residual.x(), expectedResidual.x(), 1e-6);
				EXPECT_NEAR(residual.y(), expectedResidual.y(), 1e-6);

				EXPECT_NEAR(residual.z(), expectedResidual.z(), 1e-6);
				EXPECT_NEAR(residual.w(), expectedResidual.w(), 1e-6);

			}

			/**
			 * Asserts that calculating projecting the world position to the image space results in the expected
			 * residual error.
			 */
			void assertPointCorrespondenceResidual(const Eigen::Vector3d &worldPosition, const Eigen::Vector2d &pixel,
												   Eigen::Vector4d expectedResidual) {
				assertParametricPoint(ParametricPoint::OnPoint(pixel, worldPosition), std::move(expectedResidual));
			}

			/**
			 * Asserts that calculating projecting the world position to the image space results in the expected
			 * residual error.
			 */
			void assertLineCorrespondenceResidual(Eigen::Vector3d lineOrigin, const Eigen::Vector3d &lineHeading, double
			lambda, const Eigen::Vector2d &pixel, const Eigen::Vector4d &expectedResidual) {
				assertParametricPoint(
					ParametricPoint::OnLine(pixel, std::move(lineOrigin), lineHeading, lambda),
					expectedResidual);
			}

			/**
			 * Asserts that calculating projecting the world position to the image space results in the expected
			 * residual error.
			 */
			void assertCylinderCorrespondenceResidual(Eigen::Vector3d planeOrigin, const Eigen::Vector3d &planeSideA,
													  const Eigen::Vector3d &planeSideB, double lambda, double mu,
													  double angle,
													  const Eigen::Vector2d &pixel, Eigen::Vector4d expectedResidual) {
				assertParametricPoint(ParametricPoint::OnCylinder(pixel, std::move(planeOrigin), planeSideA, planeSideB,
																  lambda, mu, angle), std::move(expectedResidual));
			}
		};

		/**
		 * Tests the point correspondence residual calculation.
		 */
		TEST_F(ResidualsTest, testPointCorrespondence) {
			Eigen::Vector3d worldPosition{0, 0, 0};
			Eigen::Vector2d pixel{960, 0};

			assertPointCorrespondenceResidual(worldPosition, pixel, {0, 0, 0, 0});

			pixel << 0, 0;
			assertPointCorrespondenceResidual(worldPosition, pixel, {-960, 0, 0, 0});

			worldPosition << 4, 20, 5;
			pixel = providentia::camera::render(translation.data(), rotation.data(),
												intrinsics, worldPosition.data());
			assertPointCorrespondenceResidual(worldPosition, pixel, {0, 0, 0, 0});

			pixel << 50, 50;
			assertPointCorrespondenceResidual(worldPosition, pixel, {-1070, -550, 0, 0});

			pixel << 1250, 650;
			assertPointCorrespondenceResidual(worldPosition, pixel, {130, 50, 0, 0});
		}


		/**
		 * Tests the line correspondence residual calculation.
		 */
		TEST_F(ResidualsTest, testLineCorrespondence) {
			Eigen::Vector3d lineOrigin{0, 0, 0};
			Eigen::Vector3d lineHeading{0, 0, 1};
			Eigen::Vector2d pixel{960, 0};
			double lambda = 0;

			assertLineCorrespondenceResidual(lineOrigin, lineHeading, lambda, pixel, {0, 0, 0, 0});

			lambda = 5;
			pixel << 960, 600;
			assertLineCorrespondenceResidual(lineOrigin, lineHeading, lambda, pixel, {0, 0, 0, 0});

			lambda = 10;
			assertLineCorrespondenceResidual(lineOrigin, lineHeading, lambda, pixel, {0, -600, 5, 0});

			lineHeading << 0, 1, 0;
			lambda = 20;
			pixel << 960, 400;
			assertLineCorrespondenceResidual(lineOrigin, lineHeading, lambda, pixel, {0, 0, 15, 0});
		}

		/**
		 * Tests the parametricPoint correspondence residual calculation.
		 */
		TEST_F(ResidualsTest, testCylinderCorrespondence) {
			Eigen::Vector3d origin{0, 0, 0};

			Eigen::Vector3d axisA = Eigen::Vector3d::UnitZ();
			Eigen::Vector3d axisB = Eigen::Vector3d::UnitX();

			assertCylinderCorrespondenceResidual(origin, axisA, axisB, 0, 0, 0, {960, 0}, {0, 0, 0, 0});
			assertCylinderCorrespondenceResidual(origin, axisA, axisB, 5, 0, 0, {960, 600}, {0, 0, 0, 0});
			assertCylinderCorrespondenceResidual(origin, axisA, axisB, 0, -8, 0, {0, 0}, {0, 0, 0, -8});

			assertCylinderCorrespondenceResidual(origin, axisA, axisB, 5, 10, 0.5 * M_PI, {0, 0}, {-960, -600, 0, 5});

		}
	}
}

