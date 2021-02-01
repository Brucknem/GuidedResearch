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
		void assertVectorsNearEqual(const Eigen::VectorXf &a, const Eigen::VectorXf &b, float maxDifference = 1e-4);

		/**
		 * @overload
		 */
		void assertVectorsNearEqual(const Eigen::Vector4f &a, float x, float y, float z, float w = 1,
									float maxDifference = 1e-4);

		/**
		 * @overload
		 */
		void assertVectorsNearEqual(const Eigen::Vector3f &a, float x, float y, float z = 1,
									float maxDifference = 1e-4);

		/**
		 * @overload
		 */
		void assertVectorsNearEqual(const Eigen::Vector2f &a, float x, float y,
									float maxDifference = 1e-4);

		/**
		 * Base class for the camera test.
		 */
		class CameraTestBase : public ::testing::Test {
		protected:
			/**
			 * Some test intrinsics.
			 */
			Eigen::Vector4f intrinsics = Eigen::Vector4f(0.05, 0.05, 1920. / 2, 1200. / 2);

			/**
			 * A test translation.
			 */
			Eigen::Vector3f translation = Eigen::Vector3f(0, -10, 5);

			/**
			 * A test camera rotation.
			 */
			Eigen::Vector3f rotation = Eigen::Vector3f(90, 0, 0);

		public:
			/**
			 * @destructor
			 */
			~CameraTestBase() override = default;
		};
	}
}


#endif //CAMERASTABILIZATION_CAMERATESTBASE_HPP
