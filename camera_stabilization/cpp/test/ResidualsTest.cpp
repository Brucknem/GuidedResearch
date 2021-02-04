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
	}
}

