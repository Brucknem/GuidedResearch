//
// Created by brucknem on 01.02.21.
//

#include "CameraTestBase.hpp"

namespace providentia {
	namespace tests {
		void
		assertVectorsNearEqual(const Eigen::VectorXd &a, const Eigen::VectorXd &b,
							   double maxDifference) {
			Eigen::VectorXd difference = a - b;
			for (int i = 0; i < difference.rows(); i++) {
				EXPECT_NEAR(difference(i), 0, maxDifference);
			}
		}

		void assertVectorsNearEqual(const Eigen::Vector4d &a, double x, double y, double z, double w,
									double maxDifference) {
			assertVectorsNearEqual(a, Eigen::Vector4d(x, y, z, w), maxDifference);
		}

		void
		assertVectorsNearEqual(const Eigen::Vector3d &a, double x, double y, double z,
							   double maxDifference) {
			assertVectorsNearEqual(Eigen::Vector4d(a(0), a(1), a(2), 1), x, y, z, 1, maxDifference);
		}

		void
		assertVectorsNearEqual(const Eigen::Vector2d &a, double x, double y, double maxDifference) {
			assertVectorsNearEqual(Eigen::Vector3d(a(0), a(1), 1), x, y, 1, maxDifference);
		}
	}
}
