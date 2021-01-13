//
// Created by brucknem on 12.01.21.
//
#include "gtest/gtest.h"
#include "TestBase.hpp"
#include "BackgroundSegmentation.hpp"
#include <iostream>

namespace providentia {
    namespace tests {
        class StabilizerTests : public TestBase {
        };

        TEST_F(StabilizerTests, testMOG2BackgroundSubtractorRuns) {
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