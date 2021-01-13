
#ifndef CAMERASTABILIZATION_DYNAMICSTABILIZATION_HPP
#define CAMERASTABILIZATION_DYNAMICSTABILIZATION_HPP


#include "opencv2/opencv.hpp"
#include "BackgroundSegmentation.hpp"
#include "FeatureDetection.hpp"
#include "FeatureMatching.hpp"

namespace providentia {
    namespace stabilization {

        /**
         * Base class for the dynamic calibration algorithms.
         */
        class DynamicStabilizerBase {
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
             * The found homography between the frame and reference frame.
             * Minimizer for the reprojection error between the frames.
             */
            cv::Mat homography;

            /**
             * The current frame warped by the found homography and with minimal reprojection error to the reference frame.
             */
            cv::cuda::GpuMat stabilizedFrame;

        public:
            /**
             * @get
             * @return The current frame stabilized with the found homography.
             */
            const cv::cuda::GpuMat &getStabilizedFrame() const;

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
        };

        /**
         * SURF feature detection and Brute Force feature matching stabilization algorithm.
         */
        class SURFBFDynamicStabilizer : public DynamicStabilizerBase {
        public:

            /**
             * @constructor
             *
             * @ref providentia::features::SurfFeatureDetector
             */
            explicit SURFBFDynamicStabilizer(double _hessianThreshold = 1000, int _nOctaves = 4,
                                             int _nOctaveLayers = 2, bool _extended = false,
                                             float _keypointsRatio = 0.01f,
                                             bool _upright = false);
        };
    }
} // namespace providentia::calibration
#endif //CAMERASTABILIZATION_DYNAMICSTABILIZATION_HPP
