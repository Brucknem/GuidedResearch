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

	}
}


#endif //CAMERASTABILIZATION_CAMERATESTBASE_HPP
