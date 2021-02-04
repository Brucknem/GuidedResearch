//
// Created by brucknem on 01.02.21.
//

#ifndef CAMERASTABILIZATION_CAMERATESTBASE_HPP
#define CAMERASTABILIZATION_CAMERATESTBASE_HPP

#include "Eigen/Dense"
#include "gtest/gtest.h"

namespace providentia {
	namespace tests {

		/**
		 * Asserts that the elements of the given vectors are not further away than the maximal difference.
		 */
		void assertVectorsNearEqual(const Eigen::VectorXd &a, const Eigen::VectorXd &b, double maxDifference = 1e-4);

		/**
		 * @overload
		 */
		void assertVectorsNearEqual(const Eigen::Vector4d &a, double x, double y, double z, double w = 1,
									double maxDifference = 1e-4);

		/**
		 * @overload
		 */
		void assertVectorsNearEqual(const Eigen::Vector3d &a, double x, double y, double z = 1,
									double maxDifference = 1e-4);

		/**
		 * @overload
		 */
		void assertVectorsNearEqual(const Eigen::Vector2d &a, double x, double y,
									double maxDifference = 1e-4);

		class CameraTestBase : public ::testing::Test {
		protected:
			Eigen::Vector2d frustumParameters{1, 1000};
			Eigen::Vector3d intrinsics{32, 1920. / 1200., 20};

			Eigen::Vector2d imageSize{1920, 1200};

			Eigen::Vector3d translation{0, -10, 5};
			Eigen::Vector3d rotation{90, 0, 0};

			/**
			 * @destructor
			 */
			~CameraTestBase() override = default;
		};
	}
}

#endif //CAMERASTABILIZATION_CAMERATESTBASE_HPP
