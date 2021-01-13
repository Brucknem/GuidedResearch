//
// Created by brucknem on 12.01.21.
//
#include "gtest/gtest.h"
#include "TestBase.hpp"
#include "FeatureDetection.hpp"
#include <iostream>

namespace providentia {
    namespace tests {
        class FeatureDetectorTests : public TestBase {
        };

        TEST_F(FeatureDetectorTests, testSURFFeatureDetectorRuns) {
            providentia::features::SurfFeatureDetector detector(1000);
            detector.detect(testImgGPU);
            std::vector<cv::KeyPoint> keypoints = detector.getKeypointsCPU();
            int keypointsSize = 3108;
            ASSERT_EQ(keypoints.size(), keypointsSize);

            std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
                if (a.pt.x == b.pt.x) {
                    return a.pt.y < b.pt.y;
                }
                return a.pt.x < b.pt.x;
            });
            EXPECT_NEAR(keypoints[0].pt.x, 13.2758, 0.01);
            EXPECT_NEAR(keypoints[0].pt.y, 387.006, 0.01);

            EXPECT_NEAR(keypoints[keypointsSize - 1].pt.x, 1895.82421875, 0.01);
            EXPECT_NEAR(keypoints[keypointsSize - 1].pt.y, 1051.9869384765625, 0.01);
        }
    }
}