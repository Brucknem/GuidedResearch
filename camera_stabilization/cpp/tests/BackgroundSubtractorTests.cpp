//
// Created by brucknem on 12.01.21.
//
#include "gtest/gtest.h"
#include "TestBase.hpp"
#include "BackgroundSegmentation.hpp"
#include <iostream>

namespace providentia {
    namespace tests {

        /**
         * Setup for the background segmentation tests.
         */
        class BackgroundSegmentationTests : public TestBase {
        };

        /**
         * Tests that the MOG2 background segmentor gives a only black mask for all the same image.
         */
        TEST_F(BackgroundSegmentationTests, testMOG2BackgroundSubtractorRuns) {
            providentia::segmentation::MOG2BackgroundSegmentor segmentor;
            for (int i = 0; i < 20; i++) {
                segmentor.segment(testImgGPU);
            }

            ASSERT_EQ(cv::countNonZero(cv::Mat(segmentor.getBackgroundMask())),
                      testImgCPU.size().width * testImgCPU.size().height);
            ASSERT_EQ(cv::countNonZero(cv::Mat(segmentor.getForegroundMask())), 0);
        }
    }
}