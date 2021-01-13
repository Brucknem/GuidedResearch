//
// Created by brucknem on 12.01.21.
//

#ifndef CAMERASTABILIZATION_FEATUREMATCHING_HPP
#define CAMERASTABILIZATION_FEATUREMATCHING_HPP

#include "opencv2/opencv.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "FeatureDetection.hpp"

namespace providentia {
    namespace features {

        /**
         * Base class for all feature matchers.
         */
        class FeatureMatcherBase : public providentia::utils::TimeMeasurable {
        protected:

            /**
             * The k nearest neighbors of features.
             */
            cv::cuda::GpuMat knnMatchesGPU;
            std::vector<std::vector<cv::DMatch>> knnMatchesCPU;

            /**
             * A vector of good matches of features.
             */
            std::vector<cv::DMatch> goodMatches;

            /**
             * The matched points of the frame and reference frame.
             */
            std::vector<cv::Point2f> frameMatchedPoints, referenceMatchedPoints;

            /**
             * The ratio threshold of good matches for the Lowe's ratio test.
             */
            float goodMatchRatioThreshold;

            /**
             * The feature detectors.
             */
            std::shared_ptr<FeatureDetectorBase> frameFeatureDetector, referenceFeatureDetector;

            /**
             * The frame with the matches drawn into.
             */
            cv::Mat drawFrame;

            /**
             * Specific implementation of the matching algorithm.
             */
            virtual void specificMatch() = 0;

        public:
            /**
             * @constructor
             *
             * @param _goodMatchRatioThreshold The ratio threshold of good matches for the Lowe's ratio test.
             */
            explicit FeatureMatcherBase(float _goodMatchRatioThreshold = 0.75f);

            /**
             * @get
             */
            const std::vector<cv::DMatch> &getGoodMatches() const;

            /**
             * @get
             */
            const std::vector<cv::Point2f> &getReferenceMatchedPoints() const;

            /**
             * @get
             */
            const std::vector<cv::Point2f> &getFrameMatchedPoints() const;

            /**
             * Matches the detected features of two frames.
             *
             * @param frameFeatureDetector The feature detector of the frame.
             * @param referenceFeatureDetector  The feature detector of the reference frame.
             */
            void match(const std::shared_ptr<FeatureDetectorBase> &_frameFeatureDetector,
                       const std::shared_ptr<FeatureDetectorBase> &_referenceFeatureDetector);

            /**
             * Horizontally stacks the frames and draws the matched features.
             */
            cv::Mat draw();
        };

        /**
         * Brute force feature matching on GPU.
         */
        class BruteForceFeatureMatcher : public FeatureMatcherBase {
        private:
            /**
             * The CUDA GPU stream used during matching.
             */
            cv::cuda::Stream stream;

            /**
             * The Brute Force matching algorithm.
             */
            cv::Ptr<cv::cuda::DescriptorMatcher> matcher;


        public:
            /**
             *
             * @param norm
             * @param _goodMatchRatioThreshold
             */
            explicit BruteForceFeatureMatcher(cv::NormTypes norm, float _goodMatchRatioThreshold = 0.95f);

            void specificMatch() override;
        };

    }
}

#endif //CAMERASTABILIZATION_FEATUREMATCHING_HPP
