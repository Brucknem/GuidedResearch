//
// Created by brucknem on 12.01.21.
//
#include "gtest/gtest.h"
#include "FeatureDetection.hpp"
#include "FeatureMatching.hpp"
#include <iostream>

TEST(SurfFeatureDetectorTest, testBasicFunctionality) {
    cv::theRNG().state = 123456789;

    providentia::features::SurfFeatureDetector detector(1000);
    cv::Mat testImg = cv::imread("tests/feature_detection_test_image.png");
    cv::cuda::GpuMat testImg_gpu;
    testImg_gpu.upload(testImg);

    detector.detect(testImg_gpu);
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

//    cv::imshow("Detected features", detector.draw());
//    cv::waitKey(1);
}

TEST(BruteForceFeatureMatcherTest, testBasicFunctionality) {
    cv::theRNG().state = 123456789;

    std::shared_ptr<providentia::features::SurfFeatureDetector> detector = std::make_shared<providentia::features::SurfFeatureDetector>(
            1000);
    cv::Mat testImg = cv::imread("tests/feature_detection_test_image.png");
    cv::cuda::GpuMat testImg_gpu;
    testImg_gpu.upload(testImg);

    detector->detect(testImg_gpu);

    providentia::features::BruteForceFeatureMatcher matcher(cv::NORM_L2);
    matcher.match(detector, detector);

    std::vector<cv::DMatch> goodMatches = matcher.getGoodMatches();

    for (const auto &goodMatch : goodMatches) {
        ASSERT_EQ(goodMatch.distance, 0);
        ASSERT_EQ(goodMatch.trainIdx, goodMatch.queryIdx);
    }

//    cv::imshow("Matched features", matcher.draw());
//    cv::waitKey(1);
}