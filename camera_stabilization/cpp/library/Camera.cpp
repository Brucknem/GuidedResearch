//
// Created by brucknem on 27.01.21.
//

#include "Camera.hpp"
#include <iostream>

namespace providentia {
    namespace camera {
#pragma region Camera

        Camera::Camera(const Eigen::Vector4f &intrinsics, const Eigen::Vector3f &translation,
                       const Eigen::Vector3f &rotation) : Camera(providentia::camera::CameraMatrix(intrinsics),
                                                                 translation, rotation) {}

        Camera::Camera(const CameraMatrix &_cameraMatrix, const Eigen::Vector3f &translation,
                       const Eigen::Vector3f &rotation) {
            cameraMatrix = providentia::camera::CameraMatrix(_cameraMatrix);
            setTranslation(translation);
            setRotation(rotation);
        }


        const CameraMatrix &Camera::getCameraMatrix() const {
            return cameraMatrix;
        }

        const Eigen::Matrix4f &Camera::getTranslation() const {
            return translation;
        }

        const Eigen::Matrix4f &Camera::getRotation() const {
            return rotation;
        }

        Eigen::Vector3f Camera::operator*(const Eigen::Vector4f &vector) {
            Eigen::Vector4f pointInCameraSpace = viewMatrix * vector;
            return cameraMatrix * pointInCameraSpace;
        }

        void Camera::setTranslation(const Eigen::Vector3f &_translation) {
            Camera::translation = Eigen::Matrix4f::Identity();
            Camera::translation.col(3).head<3>() = _translation;

            for (int i = 0; i < 3; i++) {
                assert(Camera::translation(i, 3) == _translation(i));
            }
            setViewMatrix();
        }

        void Camera::setRotation(const Eigen::Vector3f &_rotation) {
            setRotation(_rotation(0), _rotation(1), _rotation(2));
        }

        void Camera::setRotation(float x, float y, float z) {
            rotation = Eigen::Matrix4f::Identity();
            rotation(2, 2) = -1;

            rotationCalculationBuffer = Eigen::Matrix4f::Identity();

            rotationCalculationBuffer.block<3, 3>(0, 0) = Eigen::AngleAxisf(z * M_PI / 180, Eigen::Vector3f::UnitZ()).matrix();
            rotation = rotation * rotationCalculationBuffer;

            rotationCalculationBuffer.block<3, 3>(0, 0) = Eigen::AngleAxisf(-y * M_PI / 180, Eigen::Vector3f::UnitY()).matrix();
            rotation = rotation * rotationCalculationBuffer;

            rotationCalculationBuffer.block<3, 3>(0, 0) = Eigen::AngleAxisf(-x * M_PI / 180, Eigen::Vector3f::UnitX()).matrix();
            rotation = rotation * rotationCalculationBuffer;

            setViewMatrix();
        }

        const Eigen::Matrix4f &Camera::getViewMatrix() const {
            return viewMatrix;
        }

        void Camera::setViewMatrix() {
            viewMatrix = (rotation * translation).inverse();
        }
        void Camera::setTranslation(float x, float y, float z) {
        }

        std::ostream &operator<<(std::ostream &os, const Camera &obj) {
            os << "Intrinsics" << std::endl
               << obj.getCameraMatrix() << std::endl;
            os << "Translation" << std::endl
               << obj.getTranslation() << std::endl;
            os << "Rotation" << std::endl
               << obj.getRotation() << std::endl;
            os << "View" << std::endl
               << obj.getViewMatrix();
            return os;
        }

#pragma endregion Camera

#pragma region BlenderCamera

        BlenderCamera::BlenderCamera(const Eigen::Vector3f &translation, const Eigen::Vector3f &rotation) : Camera(
                                                                                                                    providentia::camera::BlenderCameraMatrix(), translation, rotation) {}

#pragma endregion BlenderCamera

    }// namespace camera
}// namespace providentia
