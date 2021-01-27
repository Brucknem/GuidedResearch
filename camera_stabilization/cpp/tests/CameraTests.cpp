//
// Created by brucknem on 12.01.21.
//
#include "gtest/gtest.h"
#include <iostream>

#include "Camera.hpp"
#include "CameraMatrix.hpp"

namespace providentia {
    namespace tests {

        /**
         * Asserts that the elements of the given vectors are not further away than the maximal difference.
         */
        void assertVectorsNearEqual(const Eigen::Vector3f &a, const Eigen::Vector3f &b, float maxDifference = 1e-4) {
            Eigen::Vector3f difference = a - b;
            for (int i = 0; i < 3; i++) {
                EXPECT_NEAR(difference(i), 0, maxDifference);
            }
        }

        void assertRotation(const providentia::camera::Camera &camera, const Eigen::Vector3f &expectedRight, const Eigen::Vector3f &expectedUp, const Eigen::Vector3f &expectedForward) {
            assertVectorsNearEqual(camera.getRotation().block<3, 1>(0, 0), expectedRight);
            assertVectorsNearEqual(camera.getRotation().block<3, 1>(0, 1), expectedUp);
            assertVectorsNearEqual(camera.getRotation().block<3, 1>(0, 2), expectedForward);
        }

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
            providentia::camera::Camera camera(intrinsics, translation, Eigen::Vector3f::Zero());
            assertRotation(camera, Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY(), -Eigen::Vector3f::UnitZ());

            camera.setRotation(90, 0, 0);
            assertRotation(camera, Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitZ(), Eigen::Vector3f::UnitY());

            camera.setRotation(0, 90, 0);
            assertRotation(camera, -Eigen::Vector3f::UnitZ(), Eigen::Vector3f::UnitY(), -Eigen::Vector3f::UnitX());

            camera.setRotation(0, 0, 90);
            assertRotation(camera, Eigen::Vector3f::UnitY(), -Eigen::Vector3f::UnitX(), -Eigen::Vector3f::UnitZ());

            camera.setRotation(-90, 0, 0);
            assertRotation(camera, Eigen::Vector3f::UnitX(), -Eigen::Vector3f::UnitZ(), -Eigen::Vector3f::UnitY());

            camera.setRotation(90, 90, 0);
            assertRotation(camera, -Eigen::Vector3f::UnitZ(), Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY());

            camera.setRotation(90, 0, 90);
            assertRotation(camera, Eigen::Vector3f::UnitY(), Eigen::Vector3f::UnitZ(), -Eigen::Vector3f::UnitX());

            camera.setRotation(0, 90, 90);
            assertRotation(camera, -Eigen::Vector3f::UnitZ(), -Eigen::Vector3f::UnitX(), -Eigen::Vector3f::UnitY());

            camera.setRotation(90, 90, 90);
            assertRotation(camera, -Eigen::Vector3f::UnitZ(), Eigen::Vector3f::UnitY(), -Eigen::Vector3f::UnitX());

            camera.setRotation(180, 0, 0);
            assertRotation(camera, Eigen::Vector3f::UnitX(), -Eigen::Vector3f::UnitY(), Eigen::Vector3f::UnitZ());

            camera.setRotation(0, 180, 0);
            assertRotation(camera, -Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY(), Eigen::Vector3f::UnitZ());

            camera.setRotation(0, 0, 180);
            assertRotation(camera, -Eigen::Vector3f::UnitX(), -Eigen::Vector3f::UnitY(), -Eigen::Vector3f::UnitZ());

            camera.setRotation(180, 180, 180);
            assertRotation(camera, Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY(), -Eigen::Vector3f::UnitZ());
        }
    }// namespace tests
}// namespace providentia