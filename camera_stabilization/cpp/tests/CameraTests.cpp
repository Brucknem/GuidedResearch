//
// Created by brucknem on 12.01.21.
//
#include "gtest/gtest.h"
#include <iostream>

#include "Camera.hpp"
#include "Intrinsics.hpp"

namespace providentia {
    namespace tests {

        /**
         * Asserts that the elements of the given vectors are not further away than the maximal difference.
         */
        void assertVectorsNearEqual(const Eigen::VectorXf &a, const Eigen::VectorXf &b, float maxDifference = 1e-4) {
            Eigen::VectorXf difference = a - b;
            for (int i = 0; i < difference.rows(); i++) {
                EXPECT_NEAR(difference(i), 0, maxDifference);
            }
        }

        void assertRotation(const providentia::camera::Camera &camera, const Eigen::Vector3f &expectedRight, const Eigen::Vector3f &expectedUp, const Eigen::Vector3f &expectedForward) {
            assertVectorsNearEqual(camera.getRotation().block<3, 1>(0, 0), expectedRight);
            assertVectorsNearEqual(camera.getRotation().block<3, 1>(0, 1), expectedUp);
            assertVectorsNearEqual(camera.getRotation().block<3, 1>(0, 2), expectedForward);
        }

        void assertTranslation(const providentia::camera::Camera &camera, const Eigen::Vector3f &expectedTranslation) {
            assertVectorsNearEqual(camera.getTranslation().block<3, 1>(0, 3), expectedTranslation);
        }

        class CameraTests : public ::testing::Test {
        protected:
            Eigen::Vector4f intrinsics = Eigen::Vector4f(0.05, 0.05, 1920. / 2, 1200. / 2);
            Eigen::Vector3f translation = Eigen::Vector3f(0, -10, 5);
            Eigen::Vector3f rotation = Eigen::Vector3f(90, 0, 0);

            Eigen::Vector4f testPoint = Eigen::Vector4f(0, 10, 0, 1);
        };

        /**
         * Tests the camera intrinsics matrix.
         */
        TEST_F(CameraTests, testCameraIntrinsics) {
            providentia::camera::Intrinsics matrix(intrinsics);

            assertVectorsNearEqual(Eigen::Vector4f(matrix.getFocalLength()(0), matrix.getFocalLength()(1), matrix.getCenter()(0), matrix.getCenter()(1)), intrinsics);
        }


        /**
         * Tests the camera rotation matrix that is built from the euler angles rotation vector.
         */
        TEST_F(CameraTests, testCameraRotationMatrix) {
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

            camera.setRotation(360, 360, 360);
            assertRotation(camera, Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY(), -Eigen::Vector3f::UnitZ());
        }

        /**
         * Tests the camera translation matrix that is built from the translation vector.
         */
        TEST_F(CameraTests, testCameraTranslationMatrix) {
            Eigen::Vector3f translation = Eigen::Vector3f::Zero();
            providentia::camera::Camera camera(intrinsics, translation, rotation);
            assertTranslation(camera, translation);

            translation = Eigen::Vector3f(10, 10, 10);
            camera.setTranslation(translation);
            assertTranslation(camera, translation);

            translation = Eigen::Vector3f(-M_PI, M_E, M_PI * M_E);
            camera.setTranslation(translation);
            assertTranslation(camera, translation);
        }
    }// namespace tests
}// namespace providentia