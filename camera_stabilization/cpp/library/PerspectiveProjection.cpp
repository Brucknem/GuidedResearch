//
// Created by brucknem on 28.01.21.
//

#include "PerspectiveProjection.hpp"

namespace providentia {
	namespace camera {
		Eigen::Vector4f normalize(const Eigen::Vector4f &vector) {
			return vector / vector(3);
		}

		PerspectiveProjection::PerspectiveProjection(float sensorWidth, float aspectRatio, float focalLength,
													 float nearPlaneDistance,
													 float farPlaneDistance) {
			setFieldOfView(sensorWidth, aspectRatio, focalLength);
			setImagePlaneSize(nearPlaneDistance, aspectRatio);
			setFrustumPlaneDistances(nearPlaneDistance, farPlaneDistance);

			updateMatrices();
		}

		void PerspectiveProjection::updateMatrices() {
			float r = (float) imagePlaneSize.x() / 2;
			float l = -r;
			float t = (float) imagePlaneSize.y() / 2;
			float b = -t;

			frustum << frustumPlaneDistances.x(), 0, 0, 0,
					0, frustumPlaneDistances.x(), 0, 0,
					0, 0, frustumPlaneDistances.x() + frustumPlaneDistances.y(), -frustumPlaneDistances.x() *
																				 frustumPlaneDistances.y(),
					0, 0, 1, 0;

			normalization << 2 / (r - l), 0, 0, -(r + l) / (r - l),
					0, 2 / (t - b), 0, -(t + b) / (t - b),
					0, 0, 2 / (frustumPlaneDistances.y() - frustumPlaneDistances.x()),
					-(frustumPlaneDistances.y() + frustumPlaneDistances.x()) /
					(frustumPlaneDistances.y() - frustumPlaneDistances.x()),
					0, 0, 0, 1;

			projection = normalization * frustum;
		}

		void PerspectiveProjection::setFrustumPlaneDistances(float nearPlaneDistance,
															 float farPlaneDistance) {
			frustumPlaneDistances = Eigen::Vector2d(nearPlaneDistance, farPlaneDistance);
		}

		void PerspectiveProjection::setFrustumPlaneDistances(float nearPlaneDistance) {
			setFrustumPlaneDistances(nearPlaneDistance, frustumPlaneDistances.y());
		}

		void PerspectiveProjection::setImagePlaneSize(float nearPlaneDistance, float aspectRatio) {
			imagePlaneSize = Eigen::Vector2d(2 * nearPlaneDistance * tan(fieldOfView.x() * 0.5), 1);
			imagePlaneSize.y() = imagePlaneSize.x() / aspectRatio;
			setFrustumPlaneDistances(nearPlaneDistance);
		}

		void PerspectiveProjection::setImagePlaneSize(float nearPlaneDistance) {
			setImagePlaneSize(nearPlaneDistance, (float) (fieldOfView.x() / fieldOfView.y()));
		}

		void PerspectiveProjection::setFieldOfView(float sensorWidth, float aspectRatio, float focalLength) {
			fieldOfView = Eigen::Vector2d(2. * atan(0.5 * (sensorWidth / focalLength)), 1);
			fieldOfView.y() = fieldOfView.x() / aspectRatio;
		}

		std::string PerspectiveProjection::toString() const {
			std::stringstream ss;
			ss << "Frustum" << std::endl << frustum << std::endl;
			ss << "Normalization" << std::endl << normalization << std::endl;
			ss << "Projection" << std::endl << projection << std::endl;
			return ss.str();
		}

		std::ostream &operator<<(std::ostream &os, const PerspectiveProjection &perspective) {
			os << perspective.toString();
			return os;
		}

		Eigen::Vector3f PerspectiveProjection::operator*(const Eigen::Vector4f &vectorInCameraSpace) {
			return normalize(projection * vectorInCameraSpace).head<3>();
		}

		Eigen::Vector4f PerspectiveProjection::toFrustum(const Eigen::Vector4f &vectorInCameraSpace) {
			return normalize(frustum * vectorInCameraSpace);
		}
	}
}
