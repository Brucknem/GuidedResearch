//
// Created by brucknem on 12.01.21.
//
#include "gtest/gtest.h"
#include "TestBase.hpp"
#include "FeatureDetection.hpp"
#include <iostream>

namespace providentia {
    namespace tests {
        class StabilizerTests : public TestBase {
        };

        TEST_F(StabilizerTests, testSURFBruteForceFeatureMatcherRuns) {
            providentia::stabilization::SURFBFDynamicStabilizer stabilizer;
            stabilizer.stabilize(testImgGPU);

            cv::Mat homography = stabilizer.getHomography();
            for (int row = 0; row < 3; row++) {
                for (int column = 0; column < 3; column++) {
                    double value = homography.at<double>(row, column);
                    if (row == column) {
                        value -= 1;
                    }
                    ASSERT_EQ(value, 0);
                }
            }

            cv::Mat warpedTestImg;
            cv::cuda::GpuMat warpedTestImgGPU;
            cv::Mat testHomography = cv::getRotationMatrix2D(cv::Point2f(400, 500), -10, 1.0);
            cv::warpAffine(testImgCPU, warpedTestImg, testHomography, testImgCPU.size(), cv::INTER_LINEAR);
            warpedTestImgGPU.upload(warpedTestImg);

            stabilizer.stabilize(warpedTestImgGPU);
            homography = stabilizer.getHomography();
            homography = cv::Mat(homography.inv());

            for (int row = 0; row < 2; row++) {
                for (int column = 0; column < 2; column++) {
                    double actual = homography.at<double>(row, column);
                    double expected = testHomography.at<double>(row, column);
                    double difference = expected - actual;
                    EXPECT_NEAR(difference, 0, 10e-4);
                }
            }

            EXPECT_NEAR(homography.at<double>(0, 2) - testHomography.at<double>(0, 2), 0, 0.1);
            EXPECT_NEAR(homography.at<double>(0, 2) - testHomography.at<double>(0, 2), 0, 0.2);

//    cv::imshow("Warped", warpedTestImg);
//    cv::imshow("Stabilized", cv::Mat(stabilizer.getStabilizedFrame()));
//    cv::waitKey(1);
        }
    }
}