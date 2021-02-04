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

			void addPerfectPointCorrespondence(const Eigen::Vector3d &pointInWorldSpace) {
				estimator->addPointCorrespondence(pointInWorldSpace, providentia::camera::render(
						translation.data(), rotation.data(),
						frustumParameters.data(), intrinsics.data(),
						imageSize.data(),
						pointInWorldSpace.data()
				));
			}
		};

		/**
		 * Tests that the optimization with an already perfect initialization converges instantly and doesn't change
		 * the parameters.
		 */
		TEST_F(CameraPoseEstimationTests, testEstimationFromOriginalTransformation) {
//			google::InitGoogleLogging("Test");
			estimator = std::make_shared<providentia::calibration::CameraPoseEstimator>(
					translation,
					rotation,
					frustumParameters,
					intrinsics,
					imageSize
			);
			addPerfectPointCorrespondence({0, 0, 0});
			addPerfectPointCorrespondence({0, 10, 0});
			addPerfectPointCorrespondence({0, 0, 5});
			addPerfectPointCorrespondence({-4, 20, 4});
			addPerfectPointCorrespondence({1, 2, 3});

			estimator->estimate();
		}

		/**
		 * Tests that the optimization with an already perfect initialization converges instantly and doesn't change
		 * the parameters.
		 */
		TEST_F(CameraPoseEstimationTests, testEstimationOnlyTranslation) {
//			google::InitGoogleLogging("Test");
//			FLAGS_logtostderr = 1;

			Eigen::Vector3d testTranslation{0, 0, 0};
			estimator = std::make_shared<providentia::calibration::CameraPoseEstimator>(
					Eigen::Vector3d{35, -100, 15},
					rotation,
					frustumParameters,
					intrinsics,
					imageSize
			);
			addPerfectPointCorrespondence({0, 0, 5});
			addPerfectPointCorrespondence({0, 10, 5});
			addPerfectPointCorrespondence({0, 30, 5});
			addPerfectPointCorrespondence({0, 50, 5});
			addPerfectPointCorrespondence({0, 70, 5});

			addPerfectPointCorrespondence({4, 10, 0});
			addPerfectPointCorrespondence({-1, 30, -3});

			estimator->estimate();

			assertVectorsNearEqual(estimator->getTranslation(), translation);
			assertVectorsNearEqual(estimator->getRotation(), rotation);

		}
	}// namespace toCameraSpace
}// namespace providentia