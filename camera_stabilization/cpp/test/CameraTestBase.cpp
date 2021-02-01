//
// Created by brucknem on 01.02.21.
//

#include "CameraTestBase.hpp"

namespace providentia {
	namespace tests {
		void
		assertVectorsNearEqual(const Eigen::VectorXf &a, const Eigen::VectorXf &b,
							   float maxDifference) {
			Eigen::VectorXf difference = a - b;
			for (int i = 0; i < difference.rows(); i++) {
				EXPECT_NEAR(difference(i), 0, maxDifference);
			}
		}

		void assertVectorsNearEqual(const Eigen::Vector4f &a, float x, float y, float z, float w,
									float maxDifference) {
			assertVectorsNearEqual(a, Eigen::Vector4f(x, y, z, w), maxDifference);
		}

		void
		assertVectorsNearEqual(const Eigen::Vector3f &a, float x, float y, float z,
							   float maxDifference) {
			assertVectorsNearEqual(Eigen::Vector4f(a(0), a(1), a(2), 1), x, y, z, 1, maxDifference);
		}

		void
		assertVectorsNearEqual(const Eigen::Vector2f &a, float x, float y, float maxDifference) {
			assertVectorsNearEqual(Eigen::Vector3f(a(0), a(1), 1), x, y, 1, maxDifference);
		}
	}
}
