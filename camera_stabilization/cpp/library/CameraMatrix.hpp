//
// Created by brucknem on 27.01.21.
//

#ifndef CAMERASTABILIZATION_CAMERAMATRIX_HPP
#define CAMERASTABILIZATION_CAMERAMATRIX_HPP

#include "Eigen/Dense"

namespace providentia {
    namespace camera {
        class CameraMatrix {
        private:
            Eigen::Vector3f calculationBuffer;

            Eigen::Matrix3f matrix;

            Eigen::Vector2f center;

            Eigen::Vector2f focalLength;

        public:
            CameraMatrix(float focalX, float focalY, int centerX, int centerY);

            explicit CameraMatrix(const Eigen::Vector4f &intrinsics);

            const Eigen::Vector2f &getCenter() const;

            const Eigen::Vector2f &getFocalLength() const;

            const Eigen::Matrix3f &getMatrix() const;

            Eigen::Vector3f operator*(const Eigen::Vector4f &vector);

            Eigen::Vector3f operator*(const Eigen::Vector3f &vector);

        };

        std::ostream &operator<<(std::ostream &os, const CameraMatrix &obj);

        class BlenderCameraMatrix : public CameraMatrix {
        public:
            BlenderCameraMatrix();
        };
    }
}


#endif //CAMERASTABILIZATION_CAMERAMATRIX_HPP
