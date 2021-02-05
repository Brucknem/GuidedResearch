#include "gtest/gtest.h"
#include <iostream>

#include "Intrinsics.hpp"
#include "CameraTestBase.hpp"
#include "CameraPoseEstimation.hpp"
#include "RenderingPipeline.hpp"

namespace providentia {
	namespace tests {

		/**
		 * Common toCameraSpace setup for the camera tests.
		 */
		class CameraPoseEstimationTests : public CameraTestBase {
		protected:

			std::shared_ptr<providentia::calibration::CameraPoseEstimator> estimator;

			/**
			 * @destructor
			 */
			~CameraPoseEstimationTests() override = default;

			void addPointCorrespondence(const Eigen::Vector3d &pointInWorldSpace) {
				estimator->addPointCorrespondence(pointInWorldSpace, providentia::camera::render(
						translation.data(), rotation.data(),
						frustumParameters.data(), intrinsics.data(),
						imageSize.data(),
						pointInWorldSpace.data()
				));
			}

			void addSomePointCorrespondences() {
				addPointCorrespondence({0, 0, 5});
				addPointCorrespondence({0, 10, 5});
				addPointCorrespondence({0, 30, 5});
				addPointCorrespondence({0, 50, 5});
				addPointCorrespondence({0, 70, 5});

				addPointCorrespondence({4, 10, 0});
				addPointCorrespondence({-1, 30, -3});
			}

			void assertEstimation(bool log = false) {
				estimator->estimate(log);
				assertVectorsNearEqual(estimator->getTranslation(), translation);
				assertVectorsNearEqual(estimator->getRotation(), rotation);
			}
		};

		/**
		 * Tests that the initial guess is half the frustum size above the mean.
		 */
		TEST_F(CameraPoseEstimationTests, testCalculateInitialGuess) {
			estimator = std::make_shared<providentia::calibration::CameraPoseEstimator>(
					frustumParameters,
					intrinsics,
					imageSize
			);

			int size = 10;
			for (int i = -size; i <= size; ++i) {
				for (int j = -size; j <= size; ++j) {
					for (int k = -size; k <= size; ++k) {
						addPointCorrespondence({i * 1., j * 1., k * 1.});
					}
				}
			}

			estimator->calculateInitialGuess();

			assertVectorsNearEqual(estimator->getTranslation(), Eigen::Vector3d{0, 0, 500.5});
			assertVectorsNearEqual(estimator->getRotation(), Eigen::Vector3d{0, 0, 0});
		}

		/**
		 * Tests that the optimization converges to the expected extrinsic parameters.
		 */
		TEST_F(CameraPoseEstimationTests, testEstimation) {
			estimator = std::make_shared<providentia::calibration::CameraPoseEstimator>(
					frustumParameters,
					intrinsics,
					imageSize
			);
			addSomePointCorrespondences();
			assertEstimation(true);
		}

	}// namespace toCameraSpace
}// namespace providentia