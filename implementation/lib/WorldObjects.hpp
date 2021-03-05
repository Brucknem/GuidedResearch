//
// Created by brucknem on 08.02.21.
//

#ifndef CAMERASTABILIZATION_WORLDOBJECTS_HPP
#define CAMERASTABILIZATION_WORLDOBJECTS_HPP

#include "Eigen/Dense"
#include <vector>
#include <memory>

namespace providentia {
	namespace calibration {

		/**
		 * A point lying in a 2-dimensional plane defined by an origin point lying in the plane and two axis.
		 */
		class ParametricPoint {
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
			 * The expected pixel location of the point.
			 */
			Eigen::Vector2d expectedPixel;

			/**
			 * The distance from the origin in the first axis.
			 */
			double lambda;

			/**
			 * The distance from the origin in the second axis.
			 */
			double mu;

		protected:
			/**
			 * @constructor
			 *
			 * @param _origin The origin of the plane.
			 * @param _axisA One side of the plane.
			 * @param _axisB Another side of the plane.
			 * @param _lambda Optional distance from the origin in the first axis.
			 * @param _mu Optional distance from the origin in the second axis.
			 */
			ParametricPoint(Eigen::Vector3d _origin, const Eigen::Vector3d &_axisA, const Eigen::Vector3d &_axisB,
							double _lambda = 0, double _mu = 0);

		public:
			/**
			 * @destructor
			 */
			virtual ~ParametricPoint() = default;

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
			double *getLambda();

			/**
			 * @get The length of the second axis.
			 */
			double *getMu();

			/**
			 * @get
			 */
			const Eigen::Vector2d &getExpectedPixel() const;

			/**
			 * @set
			 */
			void setExpectedPixel(const Eigen::Vector2d &_expectedPixel);

			/**
			 * Factory for a [x, y, z] world point.
			 *
			 * @param _expectedPixel The expected pixel.
 			 * @param _worldPosition The world position of the point.
			 */
			static ParametricPoint
			OnPoint(const Eigen::Vector2d &_expectedPixel, const Eigen::Vector3d &_worldPosition);

			/**
			 * @copydoc
			 */
			static ParametricPoint OnPoint(const Eigen::Vector3d &_worldPosition);

			/**
 			 * Factory for a [x, y, z] world point on a parametric line.
 			 *
			 * @param _expectedPixel The expected pixel.
			 * @param _origin The origin of the plane.
			 * @param _heading The heading of the line.
			 * @param _lambda Optional distance from the origin in heading direction.
			 */
			static ParametricPoint
			OnLine(const Eigen::Vector2d &_expectedPixel, Eigen::Vector3d _origin, const Eigen::Vector3d &_heading,
				   double _lambda = 0);

			/**
			 * @copydoc
			 */
			static ParametricPoint OnLine(Eigen::Vector3d _origin, const Eigen::Vector3d &_heading, double _lambda = 0);

			/**
 			 * Factory for a [x, y, z] world point on a parametric plane.
			 *
			 * @param _expectedPixel The expected pixel.
			 * @param _origin The origin of the plane.
			 * @param _axisA One side of the plane.
			 * @param _axisB Another side of the plane.
			 * @param _lambda Optional distance from the origin in the first axis.
			 * @param _mu Optional distance from the origin in the second axis.
			 */
			static ParametricPoint
			OnPlane(const Eigen::Vector2d &_expectedPixel, Eigen::Vector3d _origin, const Eigen::Vector3d &_axisA,
					const Eigen::Vector3d &_axisB,
					double _lambda = 0, double _mu = 0);

			/**
			 * @copydoc
			 */
			static ParametricPoint
			OnPlane(Eigen::Vector3d _origin, const Eigen::Vector3d &_axisA, const Eigen::Vector3d &_axisB,
					double _lambda = 0, double _mu = 0);
		};

		/**
		 * A world object containing of a set of points.
		 */
		class WorldObject {
		private:
			/**
			 * The
			 */
			std::vector<std::shared_ptr<ParametricPoint>> points;

			/**
			 * An optional id.
			 */
			std::string id;

		public:

			/**
			 * @constructor
			 */
			explicit WorldObject() = default;

			/**
			 * @constructor
			 *
			 * @param point Initializes the object with the given point.
			 */
			explicit WorldObject(const ParametricPoint &point);

			/**
			 * Adds the given point to the object.
			 */
			void add(const ParametricPoint &point);

			/**
			 * The inverse number of points.
			 */
			double getWeight() const;

			/**
			 * @get
			 */
			const std::vector<std::shared_ptr<ParametricPoint>> &getPoints() const;

			/**
			 * @get The mean of the points.
			 */
			Eigen::Vector3d getMean() const;

			/**
			 * @get
			 */
			const std::string &getId() const;

			/**
			 * @set
			 */
			void setId(const std::string &id);
		};
	}
}

#endif //CAMERASTABILIZATION_WORLDOBJECTS_HPP
