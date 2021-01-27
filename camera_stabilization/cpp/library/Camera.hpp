//
// Created by brucknem on 27.01.21.
//

#ifndef CAMERASTABILIZATION_CAMERA_HPP
#define CAMERASTABILIZATION_CAMERA_HPP

#include "Eigen/Dense"
#include "Intrinsics.hpp"
#include <memory>

namespace providentia {
    namespace camera {

        class Camera {
        private:
            providentia::camera::Intrinsics cameraMatrix = providentia::camera::Intrinsics(0, 0, 0, 0);

            Eigen::Matrix4f translation;

            Eigen::Matrix4f rotation, rotationCalculationBuffer;

            Eigen::Matrix4f viewMatrix, viewMatrixInverse;

            void setViewMatrix();

        public:
            Camera(const Eigen::Vector4f &intrinsics, const Eigen::Vector3f &translation,
                   const Eigen::Vector3f &rotation);

            Camera(const providentia::camera::Intrinsics &cameraMatrix, const Eigen::Vector3f &translation,
                   const Eigen::Vector3f &rotation);

            const Intrinsics &getCameraMatrix() const;

            const Eigen::Matrix4f &getTranslation() const;

            const Eigen::Matrix4f &getRotation() const;

            const Eigen::Matrix4f &getViewMatrix() const;

            void setTranslation(const Eigen::Vector3f &_translation);

            void setTranslation(float x, float y, float z);

            void setRotation(const Eigen::Vector3f &_rotation);

            void setRotation(float x, float y, float z);

            Eigen::Vector3f operator*(const Eigen::Vector4f &vector);
        };

        std::ostream &operator<<(std::ostream &os, const Camera &obj);

        class BlenderCamera : public Camera {
        public:
            BlenderCamera(const Eigen::Vector3f &translation = Eigen::Vector3f(0, -10, 5),
                          const Eigen::Vector3f &rotation = Eigen::Vector3f(76.5, 0, 0));
        };
    }// namespace camera
}// namespace providentia

#endif//CAMERASTABILIZATION_CAMERA_HPP
