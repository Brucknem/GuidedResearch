//
// Created by brucknem on 12.01.21.
//
#include "gtest/gtest.h"
#include <iostream>

#include "Camera.hpp"
#include "CameraMatrix.hpp"

namespace providentia {
    namespace tests {

        class CameraTests : public ::testing::Test {
        protected:
            Eigen::Vector4f intrinsics = Eigen::Vector4f(0.05, 0.05, 1920. / 2, 1200. / 2);
            Eigen::Vector3f translation = Eigen::Vector3f(0, -10, 5);
            Eigen::Vector3f rotation = Eigen::Vector3f(90, 0, 0);

            Eigen::Vector4f testPoint = Eigen::Vector4f(0, 10, 0, 1);
        };

        /**
         * Tests the Camera Matrix.
         */
        TEST_F(CameraTests, testCameraMatrix) {
            providentia::camera::CameraMatrix matrix(intrinsics);

            std::cout << matrix << std::endl;
            std::cout << matrix * testPoint;
        }


        /**
         * Tests the Camera.
         */
        TEST_F(CameraTests, testCameraCreation) {
            providentia::camera::Camera camera(intrinsics, translation, rotation);

            std::cout << camera;

        }
    }
}