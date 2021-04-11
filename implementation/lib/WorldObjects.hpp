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
		 * A point lying in a 2-dimensional parametricPoint defined by an origin point lying in the parametricPoint and two axis.
		 */
		class ParametricPoint {
		protected:
			/**
			 * The origin of the parametricPoint. This point lies within the parametricPoint.
			 */
			Eigen::Vector3d origin;

			/**
			 * A normalized axis of the parametricPoint.
			 */
			Eigen::Vector3d axisA;

			/**
			 * Another normalized axis of the parametricPoint.
			 */
			Eigen::Vector3d axisB;

			/**
			 * The expected pixel location of the point.
			 */
			Eigen::Vector2d expectedPixel;

			/**
			 * The distance from the origin in the first axis.
			 */
			double *lambda;

			/**
			 * The distance from the origin in the second axis.
			 */
			double *mu;

			/**
			 * Flag if the expected pixel is set.
			 */
			bool isExpectedPixelSet = false;

		protected:
			/**
			 * @constructor
			 *
			 * @param origin The origin of the parametricPoint.
			 * @param axisA One side of the parametricPoint.
			 * @param axisB Another side of the parametricPoint.
			 * @param lambda Optional distance from the origin in the first axis.
			 * @param mu Optional distance from the origin in the second axis.
			 */
			ParametricPoint(Eigen::Vector3d origin, const Eigen::Vector3d &axisA, const Eigen::Vector3d &axisB,
							double lambda = 0, double mu = 0);

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
			double *getLambda() const;

			/**
			 * @get The length of the second axis.
			 */
			double *getMu() const;

			/**
			 * @get
			 */
			const Eigen::Vector2d &getExpectedPixel() const;

			/**
			 * @set
			 */
			void setExpectedPixel(const Eigen::Vector2d &expectedPixel);

			/**
			 * @get
			 */
			bool hasExpectedPixel() const;

			/**
			 * Factory for a [x, y, z] world point.
			 *
			 * @param expectedPixel The expected pixel.
 			 * @param worldPosition The world position of the point.
			 */
			static ParametricPoint
			OnPoint(const Eigen::Vector2d &expectedPixel, const Eigen::Vector3d &worldPosition);

			/**
			 * @copydoc
			 */
			static ParametricPoint OnPoint(const Eigen::Vector3d &worldPosition);

			/**
 			 * Factory for a [x, y, z] world point on a parametric line.
 			 *
			 * @param expectedPixel The expected pixel.
			 * @param origin The origin of the parametricPoint.
			 * @param heading The heading of the line.
			 * @param lambda Optional distance from the origin in heading direction.
			 */
			static ParametricPoint
			OnLine(const Eigen::Vector2d &expectedPixel, Eigen::Vector3d origin, const Eigen::Vector3d &heading,
				   double lambda = 0);

			/**
			 * @copydoc
			 */
			static ParametricPoint OnLine(Eigen::Vector3d origin, const Eigen::Vector3d &heading, double lambda = 0);

			/**
 			 * Factory for a [x, y, z] world point on a parametric parametricPoint.
			 *
			 * @param expectedPixel The expected pixel.
			 * @param origin The origin of the parametricPoint.
			 * @param axisA One side of the parametricPoint.
			 * @param axisB Another side of the parametricPoint.
			 * @param lambda Optional distance from the origin in the first axis.
			 * @param mu Optional distance from the origin in the second axis.
			 */
			static ParametricPoint
			OnPlane(const Eigen::Vector2d &expectedPixel, Eigen::Vector3d origin, const Eigen::Vector3d &axisA,
					const Eigen::Vector3d &axisB,
					double lambda = 0, double mu = 0);

			/**
			 * @copydoc
			 */
			static ParametricPoint
			OnPlane(Eigen::Vector3d origin, const Eigen::Vector3d &axisA, const Eigen::Vector3d &axisB,
					double lambda = 0, double mu = 0);
		};

		/**
		 * A world object containing of a set of points.
		 */
		class WorldObject {
		private:
			/**
			 * The
			 */
			std::vector<ParametricPoint> points;

			/**
			 * An optional id.
			 */
			std::string id;

			double height = 0;

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
			const std::vector<ParametricPoint> &getPoints() const;

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

			/**
			 * @get
			 */
			double getHeight() const;

			/**
			 * @set
			 */
			void setHeight(double height);

			/**
			 * @get
			 */
			int getNumPoints() const;
		};
	}
}

#endif //CAMERASTABILIZATION_WORLDOBJECTS_HPP
