//
// Created by brucknem on 08.02.21.
//

#ifndef CAMERASTABILIZATION_WORLDOBJECTS_HPP
#define CAMERASTABILIZATION_WORLDOBJECTS_HPP

#include "Eigen/Dense"

namespace providentia {
	namespace calibration {

		/**
		 * A point lying in a 2-dimensional plane defined by an origin point lying in the plane and two axis.
		 */
		class PointOnPlane {
		protected:
			/**
			 * The origin of the plane. This point lies within the plane.
			 */
			Eigen::Vector3d origin;

			/**
			 * A normalized axis of the plane.
			 */
			Eigen::Vector3d axisA;

			/**
			 * Another normalized axis of the plane.
			 */
			Eigen::Vector3d axisB;

			/**
			 * The distance from the origin in the first axis.
			 */
			double lambda = 0;

			/**
			 * The distance from the origin in the second axis.
			 */
			double mu = 0;

		public:
			/**
			 * @constructor
			 *
			 * @param _origin The origin of the plane.
			 * @param _axisA One side of the plane.
			 * @param _axisB Another side of the plane.
			 * @param _lambda Optional distance from the origin in the first axis.
			 * @param _mu Optional distance from the origin in the second axis.
			 */
			PointOnPlane(Eigen::Vector3d _origin, const Eigen::Vector3d &_axisA, const Eigen::Vector3d &_axisB, double
			_lambda = 0, double _mu = 0);

			/**
			 * @destructor
			 */
			virtual ~PointOnPlane() = default;

			/**
			 * @get The world position of the point.
			 */
			Eigen::Vector3d getPosition() const;

			/**
			 * @get The world position of the origin.
			 */
			const Eigen::Vector3d &getOrigin() const;

			/**
			 * @get The world direction of the first axis.
			 */
			const Eigen::Vector3d &getAxisA() const;

			/**
			 * @get The world direction of the second axis.
			 */
			const Eigen::Vector3d &getAxisB() const;

			/**
			 * @get The length of the first axis.
			 */
			const double *getLambda() const;

			/**
			 * @get The length of the second axis.
			 */
			const double *getMu() const;
		};

		/**
		 * A point lying in a 1-dimensional line defined by an origin point lying on the line and the lines heading.
		 * A point on a line is a degenerated point on a plane with one side begin the zero vector.
		 */
		class PointOnLine : public PointOnPlane {
		public:
			/**
			 * @constructor
			 *
			 * @param _origin The origin of the plane.
			 * @param _heading The heading of the line.
			 * @param _lambda Optional distance from the origin in heading direction.
			 */
			PointOnLine(Eigen::Vector3d _origin, const Eigen::Vector3d &_heading, double _lambda = 0);

			/**
			 * @destructor
			 */
			~PointOnLine() override = default;
		};

		/**
		 * A point in world space.
		 * A point is a degenerated point on a line with the heading being the zero vector.
		 */
		class Point : public PointOnLine {
		public:
			/**
			 * @constructor
			 *
			 * @param _worldPosition The world position of the point.
			 */
			explicit Point(Eigen::Vector3d _worldPosition);

			/**
			 * @destructor
			 */
			~Point() override = default;
		};
	}
}

#endif //CAMERASTABILIZATION_WORLDOBJECTS_HPP
