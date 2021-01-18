
#ifndef CAMERASTABILIZATION_DYNAMICSTABILIZATION_HPP
#define CAMERASTABILIZATION_DYNAMICSTABILIZATION_HPP


#include "opencv2/opencv.hpp"
#include "BackgroundSegmentation.hpp"
#include "FeatureDetection.hpp"
#include "FeatureMatching.hpp"
#include "FrameWarping.hpp"

namespace providentia {
    namespace stabilization {

        /**
         * Base class for the dynamic calibration algorithms.
         */
        class DynamicStabilizerBase : public providentia::utils::TimeMeasurable {
        private:
            /**
             * The warmup iterations before the keyframe may be updated.
             */
            int warmUp = 50;

            /**
             * The current iteration
             */
            int currentIteration = 0;

        protected:
            /**
             * Feature detectors for the current frame and reference frame.
             */
            std::shared_ptr<providentia::features::FeatureDetectorBase> frameFeatureDetector, referenceFeatureDetector;

            /**
             * Feature matcher to match the frame and reference frame features.
             */
            std::shared_ptr<providentia::features::FeatureMatcherBase> matcher;

            /**
             * Warps the frame based on the given matches.
             */
            std::shared_ptr<providentia::stabilization::FrameWarper> warper;

            /**
             * Generates the foreground background masks.
             */
            std::shared_ptr<providentia::segmentation::BackgroundSegmentorBase> segmentor;

            /**
             * @constructor
             */
            DynamicStabilizerBase();

        public:
            /**
             * @destructor
             */
            virtual ~DynamicStabilizerBase();

            /**
             * @get The warper used to align the frame with the reference frame.
             */
            const std::shared_ptr<providentia::stabilization::FrameWarper> &getWarper() const;

            /**
             * @get The background segmentor used to mask the frames.
             */
            const std::shared_ptr<providentia::segmentation::BackgroundSegmentorBase> &getSegmentor() const;

            /**
             * @get The feature detector for the current frame.
             */
            const std::shared_ptr<providentia::features::FeatureDetectorBase> &getFrameFeatureDetector() const;

            /**
             * @get The matcher used for matching the frame and reference frame.
             */
            const std::shared_ptr<providentia::features::FeatureMatcherBase> &getMatcher() const;

            /**
             * @get The current frame stabilized with the found homography.
             */
            const cv::cuda::GpuMat &getStabilizedFrame() const;

            /**
             * @get
             * @return The reference frame.
             */
            const cv::cuda::GpuMat &getReferenceframe() const;

            /**
             * @get
             * @return The reference frame mask.
             */
            const cv::cuda::GpuMat &getReferenceframeMask() const;

            /**
             * @get
             * @return The found homography minimizing the reprojection error between the frame and reference frame.
             */
            const cv::Mat &getHomography() const;

            /**
             * Main algorithm. <br>
             * 1. Detects features in the current frame and reference frame. <br>
             * 2. Matches features of the frames. <br>
             * 3. Finds homography by minimizing the reprojection error. <br>
             * 4. Warps the frame using the found homography. <br>
             *
             * @param _frame The frame to stabilize.
             */
            void stabilize(const cv::cuda::GpuMat &_frame);

            /**
             * Draws the original and the stabilized frame aside.
             */
            cv::Mat draw();

            /**
             * Updates the keyframe.
             */
            void updateKeyframe();
        };

        /**
         * SURF feature detection and Brute Force feature matching stabilization algorithm.
         */
        class SURFBFDynamicStabilizer : public DynamicStabilizerBase {
        public:

            /**
             * @constructor
             */
            explicit SURFBFDynamicStabilizer(double _hessianThreshold = 1000, int _nOctaves = 4,
                                             int _nOctaveLayers = 2, bool _extended = false,
                                             float _keypointsRatio = 0.01f,
                                             bool _upright = false);

            /**
             * @destructor
             */
            ~SURFBFDynamicStabilizer() override;
        };

        /**
         * ORB feature detection and Brute Force feature matching stabilization algorithm.
         */
        class ORBBFDynamicStabilizer : public DynamicStabilizerBase {
        public:

            /**
             * @constructor
             */
            explicit ORBBFDynamicStabilizer(int nfeatures = 1e4,
                                            float scaleFactor = 1.2f,
                                            int nlevels = 8,
                                            int edgeThreshold = 31,
                                            int firstLevel = 0,
                                            int WTA_K = 2,
                                            int scoreType = cv::ORB::FAST_SCORE,
                                            int patchSize = 31,
                                            int fastThreshold = 20,
                                            bool blurForDescriptor = false);

            /**
             * @destructor
             */
            ~ORBBFDynamicStabilizer() override;
        };

        /**
         * Dynamic stabilization with Fast feature detectors, FREAK feature descriptors and Brute Force matching.
         */
        class FastFREAKBFDynamicStabilizer : public DynamicStabilizerBase {
        public:
            /**
             * @constructor
             */
            explicit FastFREAKBFDynamicStabilizer(int threshold = 50,
                                                  bool nonmaxSuppression = true,
                                                  cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16,
                                                  int max_npoints = 5000,
                                                  bool orientationNormalized = true,
                                                  bool scaleNormalized = true,
                                                  float patternScale = 22.0f,
                                                  int nOctaves = 4,
                                                  const std::vector<int> &selectedPairs = std::vector<int>());

            /**
             * @destructor
             */
            ~FastFREAKBFDynamicStabilizer() override;
        };
    }
} // namespace providentia::calibration
#endif //CAMERASTABILIZATION_DYNAMICSTABILIZATION_HPP
