//
// Created by brucknem on 04.02.21.
//

#include "gtest/gtest.h"
#include "CameraTestBase.hpp"
#include "Residuals.hpp"

#include <utility>

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

			/**
			 * Asserts that calculating projecting the world position to the image space results in the expected
			 * residual error.
			 */
			void assertPointCorrespondenceResidual(Eigen::Vector3d worldPosition, Eigen::Vector2d pixel,
												   Eigen::Vector2d expectedResidual) {
				Eigen::Vector2d residual;
				providentia::calibration::PointCorrespondenceResidual pointCorrespondenceResidual = {
						std::move(pixel), std::move(worldPosition), frustumParameters, intrinsics, imageSize
				};
				pointCorrespondenceResidual(translation.data(), rotation.data(), residual.data());

				EXPECT_NEAR(residual.x(), expectedResidual.x(), 1e-6);
				EXPECT_NEAR(residual.y(), expectedResidual.y(), 1e-6);
			}

			/**
			 * Asserts that calculating projecting the world position to the image space results in the expected
			 * residual error.
			 */
			void assertLineCorrespondenceResidual(Eigen::Vector3d lineOrigin, Eigen::Vector3d lineHeading, double
			lambda, Eigen::Vector2d pixel, Eigen::Vector2d expectedResidual) {
				Eigen::Vector2d residual;
				providentia::calibration::LineCorrespondenceResidual lineCorrespondenceResidual = {
						std::move(pixel), std::move(lineOrigin), std::move(lineHeading), frustumParameters, intrinsics,
						imageSize
				};
				lineCorrespondenceResidual(translation.data(), rotation.data(), &lambda, residual.data());

				EXPECT_NEAR(residual.x(), expectedResidual.x(), 1e-6);
				EXPECT_NEAR(residual.y(), expectedResidual.y(), 1e-6);
			}

			/**
			 * Asserts that calculating projecting the world position to the image space results in the expected
			 * residual error.
			 */
			void assertPlaneCorrespondenceResidual(Eigen::Vector3d planeOrigin, Eigen::Vector3d planeSideA,
												   double lambda, Eigen::Vector3d planeSideB, double mu,
												   Eigen::Vector2d pixel, Eigen::Vector2d expectedResidual) {
				Eigen::Vector2d residual;
				providentia::calibration::PlaneCorrespondenceResidual planeCorrespondenceResidual = {
						std::move(pixel), std::move(planeOrigin), std::move(planeSideA), std::move(planeSideB),
						frustumParameters, intrinsics,
						imageSize
				};
				planeCorrespondenceResidual(translation.data(), rotation.data(), &lambda, &mu, residual.data());

				EXPECT_NEAR(residual.x(), expectedResidual.x(), 1e-6);
				EXPECT_NEAR(residual.y(), expectedResidual.y(), 1e-6);
			}
		};

		/**
		 * Tests the point correspondence residual calculation.
		 */
		TEST_F(ResidualsTest, testPointCorrespondence) {
			Eigen::Vector3d worldPosition{0, 0, 0};
			Eigen::Vector2d pixel{960, 0};

			assertPointCorrespondenceResidual(worldPosition, pixel, {0, 0});

			pixel << 0, 0;
			assertPointCorrespondenceResidual(worldPosition, pixel, {-960, 0});

			worldPosition << 4, 20, 5;
			pixel = providentia::camera::render(translation.data(), rotation.data(), frustumParameters.data(),
												intrinsics.data(), imageSize.data(), worldPosition.data());
			assertPointCorrespondenceResidual(worldPosition, pixel, {0, 0});

			pixel << 50, 50;
			assertPointCorrespondenceResidual(worldPosition, pixel, {-1070, -550});

			pixel << 1250, 650;
			assertPointCorrespondenceResidual(worldPosition, pixel, {130, 50});
		}


		/**
		 * Tests the line correspondence residual calculation.
		 */
		TEST_F(ResidualsTest, testLineCorrespondence) {
			Eigen::Vector3d lineOrigin{0, 0, 0};
			Eigen::Vector3d lineHeading{0, 0, 1};
			Eigen::Vector2d pixel{960, 0};
			double lambda = 0;

			assertLineCorrespondenceResidual(lineOrigin, lineHeading, lambda, pixel, {0, 0});

			lambda = 5;
			pixel << 960, 600;
			assertLineCorrespondenceResidual(lineOrigin, lineHeading, lambda, pixel, {0, 0});

			lambda = 10;
			assertLineCorrespondenceResidual(lineOrigin, lineHeading, lambda, pixel, {0, -600});

			lineHeading << 0, 1, 0;
			lambda = 20;
			pixel << 960, 400;
			assertLineCorrespondenceResidual(lineOrigin, lineHeading, lambda, pixel, {0, 0});
		}

		/**
		 * Tests the plane correspondence residual calculation.
		 */
		TEST_F(ResidualsTest, testPlaneCorrespondence) {
			Eigen::Vector3d planeOrigin{0, 0, 0};

			Eigen::Vector3d planeSideA{1, 0, 0};
			Eigen::Vector3d planeSideB{0, 0, 1};

			Eigen::Vector2d pixel{960, 0};
			double lambda = 0;
			double mu = 0;

			assertPlaneCorrespondenceResidual(planeOrigin, planeSideA, lambda, planeSideB, mu, pixel, {0, 0});

			lambda = -8;
			pixel << 0, 0;
			assertPlaneCorrespondenceResidual(planeOrigin, planeSideA, lambda, planeSideB, mu, pixel, {0, 0});

			lambda = 8;
			mu = 10;
			assertPlaneCorrespondenceResidual(planeOrigin, planeSideA, lambda, planeSideB, mu, pixel, {-1920, -1200});

			lambda = 8;
			mu = 10;
			pixel << 1920, 1200;
			assertPlaneCorrespondenceResidual(planeOrigin, planeSideA, lambda, planeSideB, mu, pixel, {0, 0});
		}
	}
}

