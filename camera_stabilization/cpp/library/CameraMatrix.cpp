//
// Created by brucknem on 27.01.21.
//

#include <cassert>
#include <iostream>

#include "CameraMatrix.hpp"

namespace providentia {
    namespace camera {

#pragma region CameraMatrix

        CameraMatrix::CameraMatrix(float focalX, float focalY, int centerX, int centerY) {
            matrix << focalX, 0, centerX,
                    0, focalY, centerY,
                    0, 0, 1;

            focalLength << focalX, focalY;
            center << centerX, centerY;

            assert(getFocalLength()(0) == focalX);
            assert(getFocalLength()(1) == focalY);

            assert(getCenter()(0) == centerX);
            assert(getCenter()(1) == centerY);
        }

        CameraMatrix::CameraMatrix(const Eigen::Vector4f &intrinsics) : CameraMatrix(intrinsics(0), intrinsics(1),
                                                                                     (int) intrinsics(2),
                                                                                     (int) intrinsics(3)) {}


        const Eigen::Vector2f &CameraMatrix::getCenter() const {
            return center;
        }

        const Eigen::Vector2f &CameraMatrix::getFocalLength() const {
            return focalLength;
        }

        const Eigen::Matrix3f &CameraMatrix::getMatrix() const {
            return matrix;
        }

        Eigen::Vector3f CameraMatrix::operator*(const Eigen::Vector4f &vector) {
            calculationBuffer = vector.head<3>() / vector(3);
            std::cout << calculationBuffer << std::endl;
            return *this * calculationBuffer;
        }

        Eigen::Vector3f CameraMatrix::operator*(const Eigen::Vector3f &vector) {
            return matrix * vector;
        }

        std::ostream &operator<<(std::ostream &os, const CameraMatrix &obj) {
            os << obj.getMatrix();
            return os;
        }


#pragma endregion CameraMatrix

#pragma region BlenderCameraMatrix

        BlenderCameraMatrix::BlenderCameraMatrix() : CameraMatrix(0.05, 0.05, 1920 / 2, 1200 / 2) {}


#pragma endregion BlenderCameraMatrix

    }
}