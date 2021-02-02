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

		/**
		 * Base class for the camera test.
		 */
		class CameraTestBase : public ::testing::Test {
		protected:
			/**
			 * Some test intrinsics.
			 */
			Eigen::Vector4d intrinsics = {0.05, 0.05, 1920. / 2, 1200. / 2};

			/**
			 * A test translation.
			 */
			Eigen::Vector3d translation = {0, -10, 5};

			/**
			 * A test camera rotation.
			 */
			Eigen::Vector3d rotation = {90, 0, 0};

			/**
			 * Buffer for test points in world space.
			 */
			Eigen::Vector4d pointInWorldSpace;

			/**
			 * Buffer for test points in camera space.
			 */
			Eigen::Vector4d pointInCameraSpace;

			/**
			 * Buffer for test points in image space.
			 */
			Eigen::Vector2d pointInImageSpace;

		public:
			/**
			 * @destructor
			 */
			~CameraTestBase() override = default;
		};
	}
}


#endif //CAMERASTABILIZATION_CAMERATESTBASE_HPP
