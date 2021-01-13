//
// Created by brucknem on 13.01.21.
//

#ifndef CAMERASTABILIZATION_TESTBASE_HPP
#define CAMERASTABILIZATION_TESTBASE_HPP

#include "gtest/gtest.h"
#include "FeatureDetection.hpp"
#include "FeatureMatching.hpp"
#include "DynamicStabilization.hpp"
#include "BackgroundSegmentation.hpp"
#include <iostream>

namespace providentia {
    namespace tests {

        /**
         * Base class for all tests.
         */
        class TestBase : public ::testing::Test {
        protected:
            /**
             * The loaded test image.
             */
            cv::Mat testImgCPU;

            /**
             * The loaded test image on GPU.
             */
            cv::cuda::GpuMat testImgGPU;

            /**
             * Sets up the random number generator for deterministic tests and loads the test image.
             */
            void SetUp() override;
        };
    }
}


#endif //CAMERASTABILIZATION_TESTBASE_HPP
