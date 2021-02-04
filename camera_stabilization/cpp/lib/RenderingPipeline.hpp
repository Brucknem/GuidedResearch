//
// Created by brucknem on 04.02.21.
//

#ifndef CAMERASTABILIZATION_RENDERINGPIPELINE_HPP
#define CAMERASTABILIZATION_RENDERINGPIPELINE_HPP

#include "Eigen/Dense"
#include "opencv2/opencv.hpp"

namespace providentia {
	namespace camera {

		/**
		 * Normalizes a homogeneous vector by dividing by its W component.
		 *
		 * @tparam T double or ceres::Jet
		 * @param vector The (possibly) unnormalized [x, y, z, w] vector.
		 *
		 * @return The normalized [x/w, y/w, z/w, 1] vector.
		 */
		template<typename T>
		Eigen::Matrix<T, 4, 1> normalize(const Eigen::Matrix<T, 4, 1> &vector);

		/**
		 * Generates a camera rotation matrix from the euler angle representation.<br>
		 * Internally creates the matrices for a rotation around the X/Y/Z axis and concatenates them.<br>
		 * Assumes a camera with no rotation to have a coordinate system of [X/Y/-Z].<br>
		 *
		 * @link https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
		 *
		 * @tparam T double or ceres::Jet
		 * @param _rotation The [x, y, z] euler angle rotation.
		 *
		 * @return The complete rotation around the three axis.
		 */
		template<typename T>
		Eigen::Matrix<T, 4, 4> getCameraRotationMatrix(const T *_rotation);

		/**
		 * Transforms the given vector from world to camera space.
		 *
		 * @tparam T double or ceres::Jet
		 * @param _translation The [x, y, z] translation of the camera in world space.
		 * @param _rotation The [x, y, z] euler angle rotation of the camera around the world axis.
		 * @param vector The [x, y, z, w] vector in world space.
		 *
		 * @return The [x, y, z, w] vector in camera space.
		 */
		template<typename T>
		Eigen::Matrix<T, 4, 1> toCameraSpace(const T *_translation, const T *_rotation, const T *vector);

		/**
		 * Gets the matrix defining the (unnormalized) view frustum.
		 *
		 * @tparam T double or ceres::Jet
		 * @param _frustumParameters The [near, far] plane distances of the frustum.
		 *
		 * @return The matrix defining the view frustum.
		 */
		template<typename T>
		Eigen::Matrix<T, 4, 4> getFrustum(const T *_frustumParameters);

		/**
		 * Transforms the given vector from camera space to the view frustum.
		 *
		 * @tparam T double or ceres::Jet
		 * @param _frustumParameters The [near, far] plane distances of the frustum.
		 * @param vector The [x, y, z, w] vector in camera space.
		 *
		 * @return The [x, y, z, w] vector in the view frustum.
		 */
		template<typename T>
		Eigen::Matrix<T, 4, 1> toFrustum(const T *_frustumParameters, const T *vector);

		/**
		 * Gets the matrix that transforms vectors from camera space to clip space.
		 *
		 * @tparam T double or ceres::Jet
		 * @param _frustumParameters The [near, far] plane distances of the frustum.
		 * @param _intrinsics The [sensorWidth, aspectRatio, focalLength] defining the camera intrinsics.
		 *
		 * @return The matrix defining the clip space transformation.
		 */
		template<typename T>
		Eigen::Matrix<T, 4, 4>
		getClipSpace(const T *_frustumParameters, const T *_intrinsics);

		/**
		 * Transforms the given vectors from camera space to clip space.
		 *
		 * @tparam T double or ceres::Jet
		 * @param _frustumParameters The [near, far] plane distances of the frustum.
		 * @param _intrinsics The [sensorWidth, aspectRatio, focalLength] defining the camera intrinsics.
		 * @param vector The [x, y, z, w] vector in camera space.
		 *
		 * @return The [x, y, z, w] vector in clip space.
		 */
		template<typename T>
		Eigen::Matrix<T, 4, 1>
		toClipSpace(const T *_frustumParameters, const T *_intrinsics, const T *vector);

		/**
		 * Transforms the given vector from clip space to normalized device coordinates.
		 *
		 * @tparam T double or ceres::Jet
		 * @param vector The [x, y, z, w] vector in clip space.
		 *
		 * @return The [u, v] vector in normalized device coordinates.
		 */
		template<typename T>
		Eigen::Matrix<T, 2, 1> toNormalizedDeviceCoordinates(const T *vector);

		/**
		 * Transforms the given vector from normalized device coordinates to image space.
		 *
		 * @tparam T double or ceres::Jet
		 * @param _imageSize The [width, height] of the image.
		 * @param vector The [u, v] vector in normalized device coordinates.
		 *
		 * @return The resulting [u, v] expectedPixel location in image space.
		 */
		template<typename T>
		Eigen::Matrix<T, 2, 1> toImageSpace(const T *_imageSize, const T *vector);

		/**
		 * Wrapper for the whole rendering pipeline.
		 *
		 * @tparam T double or ceres::Jet
		 * @param _translation The [x, y, z] translation of the camera in world space.
		 * @param _rotation The [x, y, z] euler angle rotation of the camera around the world axis.
		 * @param _frustumParameters The [near, far] plane distances of the frustum.
		 * @param _intrinsics The [sensorWidth, aspectRatio, focalLength] defining the camera intrinsics.
		 * @param _imageSize The [width, height] of the image.
		 * @param vector The [x, y, z, w] vector in world space.
		 *
		 * @return The [u, v] expectedPixel location in image space.
		 */
		template<typename T>
		Eigen::Matrix<T, 2, 1>
		render(const T *_translation, const T *_rotation, const T *_frustumParameters, const T *_intrinsics,
			   const T *_imageSize, const T *vector);

		/**
		 * Renders the given vector with the given color to the given image.
		 *
		 * @tparam T double or ceres::Jet
		 * @param _translation The [x, y, z] translation of the camera in world space.
		 * @param _rotation The [x, y, z] euler angle rotation of the camera around the world axis.
		 * @param _frustumParameters The [near, far] plane distances of the frustum.
		 * @param _intrinsics The [sensorWidth, aspectRatio, focalLength] defining the camera intrinsics.
		 * @param _imageSize The [width, height] of the image.
		 * @param vector The [x, y, z, w] vector in world space.
		 * @param color The color that is assigned to the expectedPixel.
		 * @param image The image to which the expectedPixel will rendered.
		 */
		void render(const Eigen::Vector3d &_translation, const Eigen::Vector3d &_rotation,
					const Eigen::Vector2d &_frustumParameters, const Eigen::Vector3d &_intrinsics,
					const Eigen::Vector4d &vector, const cv::Vec3d &color, cv::Mat image);
	}
}

#endif //CAMERASTABILIZATION_RENDERINGPIPELINE_HPP
