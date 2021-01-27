//
// Created by brucknem on 27.01.21.
//

#include <memory>
#include "Camera.hpp"

namespace providentia {
    namespace camera {
#pragma region Camera

        Camera::Camera(const Eigen::Vector4f &intrinsics, const Eigen::Vector3f &translation,
                       const Eigen::Vector3f &rotation) : Camera(providentia::camera::CameraMatrix(intrinsics),
                                                                 translation, rotation) {}

        Camera::Camera(const CameraMatrix &cameraMatrix, const Eigen::Vector3f &translation,
                       const Eigen::Vector3f &rotation) {
            Camera::cameraMatrix = std::make_shared<providentia::camera::CameraMatrix>(cameraMatrix);

            Camera::translation = Eigen::Matrix4f::Identity();
            Camera::rotation = Eigen::Matrix4f::Identity();

            Camera::translation.col(3).head<3>() = translation;
            Camera::rotation.block<3, 3>(0, 0) = (
                    Eigen::AngleAxisf(rotation(0) * M_PI / 180, Eigen::Vector3f::UnitX()) *
                    Eigen::AngleAxisf(rotation(1) * M_PI / 180, Eigen::Vector3f::UnitY()) *
                    Eigen::AngleAxisf(rotation(2) * M_PI / 180, Eigen::Vector3f::UnitZ())).matrix();

            for (int i = 0; i < 3; i++) {
                assert(Camera::translation(i, 3) == translation(i));
            }
        }


        const CameraMatrix &Camera::getCameraMatrix() const {
            return *cameraMatrix;
        }

        const Eigen::Matrix4f &Camera::getTranslation() const {
            return translation;
        }

        const Eigen::Matrix4f &Camera::getRotation() const {
            return rotation;
        }

        std::ostream &operator<<(std::ostream &os, const Camera &obj) {
            os << "Intrinsics" << std::endl << obj.getCameraMatrix() << std::endl;
            os << "Translation" << std::endl << obj.getTranslation() << std::endl;
            os << "Rotation" << std::endl << obj.getRotation();
            return os;
        }

#pragma endregion Camera

#pragma region BlenderCamera

        BlenderCamera::BlenderCamera(const Eigen::Vector3f &translation, const Eigen::Vector3f &rotation) : Camera(
                providentia::camera::BlenderCameraMatrix(), translation, rotation) {}

#pragma endregion BlenderCamera

    }
}
