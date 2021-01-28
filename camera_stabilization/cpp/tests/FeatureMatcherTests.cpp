//
// Created by brucknem on 12.01.21.
//
#include "gtest/gtest.h"
#include "ImageTestBase.hpp"
#include "FeatureDetection.hpp"
#include <iostream>

namespace providentia {
	namespace tests {

		/**
		 * Setup for the feature matcher tests.
		 */
		class FeatureMatcherTests : public ImageTestBase {
		public:

			/**
			 * @destructor
			 */
			~FeatureMatcherTests() override = default;

			/**
			 * Asserts that the given matcher can match two identical sets of features perfectly.
			 */
			void assertMatcher(features::FeatureMatcherBase *matcher) {
				std::shared_ptr<providentia::features::SURFFeatureDetector> detector = std::make_shared<providentia::features::SURFFeatureDetector>(
						1000);
				detector->detect(testImgGPU);


				matcher->match(detector, detector);

				std::vector<cv::DMatch> goodMatches = matcher->getGoodMatches();

				for (const auto &goodMatch : goodMatches) {
					ASSERT_EQ(goodMatch.distance, 0);
					ASSERT_EQ(goodMatch.trainIdx, goodMatch.queryIdx);
				}
			}
		};

		/**
		 * Tests the Brute Force feature matcher.
		 */
		TEST_F(FeatureMatcherTests, testBruteForceFeatureMatcherRuns) {
			assertMatcher(new providentia::features::BruteForceFeatureMatcher(cv::NORM_L2));
		}

		/**
		 * Tests the Flann feature matcher
		 */
		TEST_F(FeatureMatcherTests, testFlannFeatureMatcherRuns) {
			assertMatcher(new providentia::features::FlannFeatureMatcher());
		}
	}
}