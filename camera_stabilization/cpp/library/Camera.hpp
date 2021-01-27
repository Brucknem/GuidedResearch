//
// Created by brucknem on 27.01.21.
//

#ifndef CAMERASTABILIZATION_CAMERA_HPP
#define CAMERASTABILIZATION_CAMERA_HPP

#include "CameraMatrix.hpp"
#include "Eigen/Dense"

namespace providentia {
    namespace camera {

        class Camera {
        private:
            std::shared_ptr<providentia::camera::CameraMatrix> cameraMatrix;

            Eigen::Matrix4f translation;

            Eigen::Matrix4f rotation;

        public:
            Camera(const Eigen::Vector4f &intrinsics, const Eigen::Vector3f &translation,
                   const Eigen::Vector3f &rotation);

            Camera(const providentia::camera::CameraMatrix &cameraMatrix, const Eigen::Vector3f &translation,
                   const Eigen::Vector3f &rotation);

            const CameraMatrix &getCameraMatrix() const;

            const Eigen::Matrix4f &getTranslation() const;

            const Eigen::Matrix4f &getRotation() const;
        };

        std::ostream &operator<<(std::ostream &os, const Camera &obj);

        class BlenderCamera : public Camera {
        public:
            BlenderCamera(const Eigen::Vector3f &translation = Eigen::Vector3f(0, -10, 5),
                          const Eigen::Vector3f &rotation = Eigen::Vector3f(76.5, 0, 0));
        };
    }
}

#endif //CAMERASTABILIZATION_CAMERA_HPP
