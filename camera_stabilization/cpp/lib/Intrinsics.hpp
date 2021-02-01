
#ifndef CAMERASTABILIZATION_INTRINSICS_HPP
#define CAMERASTABILIZATION_INTRINSICS_HPP

#include "Eigen/Dense"

// TODO https://www.mathworks.com/help/vision/ug/camera-calibration.html

namespace providentia {
	namespace camera {

		/**
		 * The camera intrinsic parameters that are used to project vectors from
		 * camera space to image space.
		 */
		class Intrinsics {
		private:
			/**
			 * A buffer used during calculations.
			 */
			Eigen::Vector3f calculationBuffer;

			/**
			 * The actual intrinsics matrix.
			 */
			Eigen::Matrix3f matrix;

			/**
			 * The camera center point.
			 */
			Eigen::Vector2f center;

			/**
			 * The focal lengths.
			 */
			Eigen::Vector2f focalLength;

		public:

			/**
			 * @constructor
			 *
			 * @param focalX The focal length in X direction.
			 * @param focalY The focal length in Y direction.
			 * @param centerX The X pixel coordinate of the image center (principal point).
			 * @param centerY The Y pixel coordinate of the image center (principal point).
			 */
			Intrinsics(float focalX, float focalY, int centerX, int centerY);

			/**
			 * @constructor
			 *
			 * @param intrinsics A vector containing the focal lengths and center point.
			 */
			explicit Intrinsics(const Eigen::Vector4f &intrinsics);

			/**
			 * @destructor
			 */
			virtual ~Intrinsics() = default;

			/**
			 * @get The center point.
			 */
			const Eigen::Vector2f &getCenter() const;

			/**
			 * @get The focal lengths.
			 */
			const Eigen::Vector2f &getFocalLength() const;

			/**
			 * @get The intrinsics matrix.
			 */
			const Eigen::Matrix3f &getMatrix() const;

			/**
			 * Multiply the given vector and project to camera space.
			 */
			Eigen::Vector3f operator*(const Eigen::Vector4f &vector);

			/**
			 * Multiply the given vector and project to camera space.
			 */
			Eigen::Vector3f operator*(const Eigen::Vector3f &vector);

			/**
			 * Adds a string representation of the intrinsics to the given stream.
			 */
			friend std::ostream &operator<<(std::ostream &os, const Intrinsics &obj);
		};


		/**
		 * Mock class for the intrinsics used in the blender test setup.
		 */
		class BlenderIntrinsics : public Intrinsics {
		public:
			/**
			 * @constructor
			 */
			BlenderIntrinsics();

			/**
			 * @destructor
			 */
			~BlenderIntrinsics() override = default;
		};
	}// namespace camera
}// namespace providentia


#endif//CAMERASTABILIZATION_INTRINSICS_HPP
