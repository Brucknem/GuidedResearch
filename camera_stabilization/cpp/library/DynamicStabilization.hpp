
#ifndef CAMERASTABILIZATION_DYNAMICSTABILIZATION_HPP
#define CAMERASTABILIZATION_DYNAMICSTABILIZATION_HPP


#include <cstdio>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <utility>
#include <opencv2/video.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include <opencv2/cudafeatures2d.hpp>
#include "opencv2/calib3d.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>
#include <thread>
#include <exception>
#include <stdexcept>
#include "BackgroundSegmentation.h"
#include "FeatureDetection.hpp"
#include "FeatureMatching.hpp"

namespace providentia {
    namespace stabilization {

        /**
         * Base class for the dynamic calibration algorithms.
         */
        class DynamicStabilizerBase {
        protected:
            std::shared_ptr<providentia::features::FeatureDetectorBase> frameFeatureDetector, referenceFeatureDetector;
            std::shared_ptr<providentia::features::FeatureMatcherBase> matcher;
        public:
            const cv::Mat &getHomography() const;

        protected:
            cv::Mat homography;
            cv::cuda::GpuMat stabilizedFrame;
        public:
            const cv::cuda::GpuMat &getStabilizedFrame() const;

        public:
            void stabilize(const cv::cuda::GpuMat &_frame);
        };

        class SURFBFDynamicStabilizer : public DynamicStabilizerBase {
        public:
            explicit SURFBFDynamicStabilizer(double _hessianThreshold = 1000, int _nOctaves = 4,
                                             int _nOctaveLayers = 2, bool _extended = false,
                                             float _keypointsRatio = 0.01f,
                                             bool _upright = false);
        };
    }
} // namespace providentia::calibration
#endif //CAMERASTABILIZATION_DYNAMICSTABILIZATION_HPP
