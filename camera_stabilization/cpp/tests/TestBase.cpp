//
// Created by brucknem on 13.01.21.
//

#include "TestBase.hpp"

void providentia::tests::TestBase::SetUp() {
    Test::SetUp();
    cv::theRNG().state = 123456789;
    testImgCPU = cv::imread("tests/feature_detection_test_image.png");
    testImgGPU.upload(testImgCPU);
}

