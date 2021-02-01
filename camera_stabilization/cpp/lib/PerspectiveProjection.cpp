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
			setFrustumPlaneDistances(nearPlaneDistance, farPlaneDistance);
			setImagePlaneBounds(nearPlaneDistance, aspectRatio);

			updateMatrices();
		}

		void PerspectiveProjection::updateMatrices() {
			frustum << frustumPlaneDistances.x(), 0, 0, 0,
					0, frustumPlaneDistances.x(), 0, 0,
					0, 0, frustumPlaneDistances.x() + frustumPlaneDistances.y(), -frustumPlaneDistances.x() *
																				 frustumPlaneDistances.y(),
					0, 0, 1, 0;

			normalization << 2 / (topRight.x() - lowerLeft.x()), 0, 0, -(topRight.x() + lowerLeft.x()) /
																	   (topRight.x() - lowerLeft.x()),
					0, 2 / (topRight.y() - lowerLeft.y()), 0, -(topRight.y() + lowerLeft.y()) /
															  (topRight.y() - lowerLeft.y()),
					0, 0, 2 / (frustumPlaneDistances.y() - frustumPlaneDistances.x()),
					-(frustumPlaneDistances.y() + frustumPlaneDistances.x()) /
					(frustumPlaneDistances.y() - frustumPlaneDistances.x()),
					0, 0, 0, 1;

			projection = normalization * frustum;
		}

		void PerspectiveProjection::setFrustumPlaneDistances(float nearPlaneDistance,
															 float farPlaneDistance) {
			frustumPlaneDistances = {nearPlaneDistance, farPlaneDistance};
		}

		void PerspectiveProjection::setImagePlaneBounds(float nearPlaneDistance, float aspectRatio) {
			topRight = {nearPlaneDistance * tan(fieldOfView.x() * 0.5), 1};
			topRight.y() = topRight.x() / aspectRatio;
			lowerLeft = -topRight;
		}

		void PerspectiveProjection::setFieldOfView(float sensorWidth, float aspectRatio, float focalLength) {
			fieldOfView = {2. * atan(0.5 * (sensorWidth / focalLength)), 1};
			fieldOfView.y() = fieldOfView.x() / aspectRatio;
		}

		std::ostream &operator<<(std::ostream &os, const PerspectiveProjection &perspective) {
			os << "Frustum" << std::endl << perspective.frustum << std::endl;
			os << "Normalization" << std::endl << perspective.normalization << std::endl;
			os << "Projection" << std::endl << perspective.projection << std::endl;
			return os;
		}

		Eigen::Vector2f PerspectiveProjection::operator*(const Eigen::Vector4f &vectorInCameraSpace) {
			return toClipSpace(vectorInCameraSpace).head<2>();
		}

		Eigen::Vector4f PerspectiveProjection::toFrustum(const Eigen::Vector4f &vectorInCameraSpace) {
			return normalize(frustum * vectorInCameraSpace);
		}

		Eigen::Vector3f PerspectiveProjection::toClipSpace(const Eigen::Vector4f &vectorInCameraSpace) {
			return normalize(projection * vectorInCameraSpace).head<3>();
		}

		const Eigen::Vector2f &PerspectiveProjection::getTopRight() const {
			return topRight;
		}

		const Eigen::Vector2f &PerspectiveProjection::getLowerLeft() const {
			return lowerLeft;
		}

		const Eigen::Vector2d &PerspectiveProjection::getFrustumPlaneDistances() const {
			return frustumPlaneDistances;
		}
	}
}
