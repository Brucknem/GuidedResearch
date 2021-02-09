#include "gtest/gtest.h"
#include <iostream>
#include <utility>

#include "Intrinsics.hpp"
#include "CameraTestBase.hpp"
#include "CameraPoseEstimation.hpp"
#include "RenderingPipeline.hpp"

using namespace providentia::calibration;

namespace providentia {
	namespace tests {

		/**
		 * Common setup for the camera tests.
		 */
		class CameraPoseEstimationTests : public CameraTestBase {
		protected:

			std::shared_ptr<providentia::calibration::CameraPoseEstimator> estimator;

			/**
			 * @destructor
			 */
			~CameraPoseEstimationTests() override = default;

			Eigen::Vector2d getPixel(const providentia::calibration::ParametricPoint &object) {
				return getPixel(object.getPosition());
			}

			Eigen::Vector2d getPixel(Eigen::Vector3d vector) {
				return providentia::camera::render(
						translation.data(), rotation.data(),
						frustumParameters.data(), intrinsics.data(),
						imageSize.data(),
						vector.data());
			}

			void addPointCorrespondence(const Eigen::Vector3d &pointInWorldSpace) {
				WorldObject worldObject(ParametricPoint::OnPoint(getPixel(pointInWorldSpace), pointInWorldSpace));
				estimator->addWorldObject(worldObject);
			}

			void addSomePointCorrespondences() {
				addPointCorrespondence({1, 2, 3});
				addPointCorrespondence({-1, 12, 13});
				addPointCorrespondence({13, 22, -4});
			}

			void assertEstimation(bool log = false) {
				estimator->estimate(log);
				assertVectorsNearEqual(estimator->getTranslation(), translation);
				assertVectorsNearEqual(estimator->getRotation(), rotation);

				if (log) {
					std::cout << "Translation" << std::endl;
					std::cout << estimator->getTranslation() << std::endl;
					std::cout << "Rotation" << std::endl;
					std::cout << estimator->getRotation() << std::endl;

					std::cout << "World worldObjects" << std::endl;
					for (const auto &worldObject : estimator->getWorldObjects()) {
						std::cout << "Next world object" << std::endl;
						for (const auto &point : worldObject.getPoints()) {
							std::cout << point->getPosition() << std::endl << std::endl;
						}
					}
				}
			}

			void addPost(const Eigen::Vector3d &origin) {
				int number = 5;
				double height = 1.5;
				WorldObject post;
				for (int i = 0; i < 5; ++i) {
					ParametricPoint point = ParametricPoint::OnLine(origin, {0, 0, 1}, (height / number) * i);
					point.setExpectedPixel(getPixel(point));
					post.add(point);
				}
				estimator->addWorldObject(post);
			}

			void addPlane(int samples) {
				Eigen::Vector3d origin(0, 0, 0);
				Eigen::Vector3d axisA(1, 0, 0);
				Eigen::Vector3d axisB(0, 1, 0);
				WorldObject plane;
				for (int i = 0; i < samples; ++i) {
					providentia::calibration::ParametricPoint point = ParametricPoint::OnPlane(
							origin,
							axisA,
							axisB,
							(rand() % 2000) / 100. - 10,
							(rand() % 2000) / 100.
					);
					point.setExpectedPixel(getPixel(point.getPosition()));
					plane.add(point);
				}
				estimator->addWorldObject(plane);
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
		TEST_F(CameraPoseEstimationTests, testEstimationOnlyWorldPositions) {
			estimator = std::make_shared<providentia::calibration::CameraPoseEstimator>(
					frustumParameters,
					intrinsics,
					imageSize
			);
			addSomePointCorrespondences();
//			estimator->setInitialGuess({0, 0, 500}, {0, 0, 0});
			assertEstimation(true);
		}


		/**
		 * Tests that the optimization converges to the expected extrinsic parameters.
		 */
		TEST_F(CameraPoseEstimationTests, testEstimationPlaneAndLines) {
			estimator = std::make_shared<providentia::calibration::CameraPoseEstimator>(
					frustumParameters,
					intrinsics,
					imageSize
			);

			Eigen::Vector3d origin, axisA, axisB;
			origin << 0, 0, 0;
			axisA << 1, 0, 0;
			axisB << 0, 1, 0;

			addPointCorrespondence({1, 2, 3});
//			addPointCorrespondence({-1, 12, 13});
//			addPointCorrespondence({13, 22, -4});

			addPost({-7.5, 10, 0});
//			addPost({7.5, 20, 0});
//
//			addPlane(100);

//			for (int i = 0; i < 20; ++i) {
//				addPost({
//								(rand() % 2000) / 100. - 10,
//								(rand() % 2000) / 100.,
//								(rand() % 2000) / 100.
//						});
//			}
//
			for (int i = 0; i < 200; i += 20) {
				addPost({-7.5, i, 0});
				addPost({7.5, i, 0});
			}

			estimator->setInitialGuess({0, -50, 0}, {80, 10, -10});
			assertEstimation(true);

		}
	}// namespace toCameraSpace
}// namespace providentia