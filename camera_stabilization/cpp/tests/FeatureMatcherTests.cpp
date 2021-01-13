//
// Created by brucknem on 12.01.21.
//
#include "gtest/gtest.h"
#include "TestBase.hpp"
#include "FeatureDetection.hpp"
#include <iostream>

namespace providentia {
    namespace tests {
        class FeatureMatcherTests : public TestBase {
        };

        TEST_F(FeatureMatcherTests, testBruteForceFeatureMatcherRuns) {
            std::shared_ptr<providentia::features::SURFFeatureDetector> detector = std::make_shared<providentia::features::SURFFeatureDetector>(
                    1000);
            detector->detect(testImgGPU);

            providentia::features::BruteForceFeatureMatcher matcher(cv::NORM_L2);
            matcher.match(detector, detector);

            std::vector<cv::DMatch> goodMatches = matcher.getGoodMatches();

            for (const auto &goodMatch : goodMatches) {
                ASSERT_EQ(goodMatch.distance, 0);
                ASSERT_EQ(goodMatch.trainIdx, goodMatch.queryIdx);
            }
        }
    }
}