//
// Created by brucknem on 27.01.21.
//

#include <cassert>
#include <iostream>

#include "Intrinsics.hpp"

namespace providentia {
    namespace camera {

#pragma region CameraMatrix

        Intrinsics::Intrinsics(float focalX, float focalY, int centerX, int centerY) {
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

        Intrinsics::Intrinsics(const Eigen::Vector4f &intrinsics) : Intrinsics(intrinsics(0), intrinsics(1),
                                                                               (int) intrinsics(2),
                                                                               (int) intrinsics(3)) {}


        const Eigen::Vector2f &Intrinsics::getCenter() const {
            return center;
        }

        const Eigen::Vector2f &Intrinsics::getFocalLength() const {
            return focalLength;
        }

        const Eigen::Matrix3f &Intrinsics::getMatrix() const {
            return matrix;
        }

        Eigen::Vector3f Intrinsics::operator*(const Eigen::Vector4f &vector) {
            calculationBuffer = vector.head<3>() / vector(3);
            std::cout << calculationBuffer << std::endl;
            return *this * calculationBuffer;
        }

        Eigen::Vector3f Intrinsics::operator*(const Eigen::Vector3f &vector) {
            return matrix * vector;
        }

        std::ostream &operator<<(std::ostream &os, const Intrinsics &obj) {
            os << obj.getMatrix();
            return os;
        }


#pragma endregion CameraMatrix

#pragma region BlenderCameraMatrix

        BlenderCameraMatrix::BlenderCameraMatrix() : Intrinsics(0.05, 0.05, 1920 / 2, 1200 / 2) {}


#pragma endregion BlenderCameraMatrix

    }// namespace camera
}// namespace providentia