//
// Created by brucknem on 12.01.21.
//

#ifndef CAMERASTABILIZATION_FEATUREDETECTION_HPP
#define CAMERASTABILIZATION_FEATUREDETECTION_HPP


#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d/cuda.hpp"
#include <opencv2/cudafeatures2d.hpp>

#include "Utils.hpp"

namespace providentia {
    namespace features {

        /**
         * Base class for all feature detectors.
         */
        class FeatureDetectorBase : public providentia::utils::TimeMeasurable {
        protected:

            /**
             * Flag if the latest mask should be used.
             */
            bool useLatestMask = false;

            /**
             * The current processed frame.
             */
            cv::cuda::GpuMat frame;

            /**
             * The current original frame.
             */
            cv::cuda::GpuMat originalFrame;

            /**
             * The masks used during processing.
             */
            cv::cuda::GpuMat latestMask, noMask, currentMask;

            /**
             * The feature descriptorsGPU.
             */
            cv::cuda::GpuMat descriptorsGPU;

            /**
             * The GPU feature keypoints.
             */
            cv::cuda::GpuMat keypointsGPU;

            /**
             * The CPU feature keypoints.
             */
            std::vector<cv::KeyPoint> keypointsCPU;

        protected:

            /**
             * The frame with the features drawn into.
             */
            cv::Mat drawFrame;

            /**
             * Specific detection implementation.
             */
            virtual void specificDetect() = 0;

            /**
             * Sets the current mask from the latest or an empty mask based on the useLatestMask flag.
             * @see FeatureDetectorBase#useLatestMask
             */
            void setCurrentMask(cv::Size _size = cv::Size());

            /**
             * @constructor
             */
            FeatureDetectorBase();

        public:

            /**
             * @get Gets the current mask from the latest or an empty mask based on the useLatestMask flag.
             */
            const cv::cuda::GpuMat &getCurrentMask(cv::Size _size = cv::Size());

            /**
             * @get
             */
            const cv::cuda::GpuMat &getOriginalFrame() const;

            /**
             * @get
             */
            const cv::cuda::GpuMat &getDescriptorsGPU() const;

            /**
             * @get
             */
            const std::vector<cv::KeyPoint> &getKeypointsCPU() const;

            /**
             * Flag if there is a frame and corresponding keypoints and descriptors present.
             *
             * @return true if a detection has been performed previously, false else.
             */
            bool isEmpty();

            /**
             * Grayscales the given frame.
             * Detects keypoints and features using the subclass specific implementations.
             *
             * @param _frame The frame used for detection.
             */
            void detect(const cv::cuda::GpuMat &_frame);

            /**
             * Detects keypoints and features using the latest mask or without a mask.
             *
             * @param _frame The frame used for detection.
             * @param _useLatestMask Flag whether or not to use the latest mask.
             *
             * @see FeatureDetectorBase#useLatestMask
             * @see FeatureDetectorBase#detect(cv::cuda::GpuMat)
             */
            void detect(const cv::cuda::GpuMat &_frame, bool _useLatestMask);

            /**
             * Detects keypoints and features using the given mask.
             *
             * @param _frame The frame used for detection.
             * @param _mask The mask used for detection.
             *
             * @see FeatureDetectorBase#detect(cv::cuda::GpuMat)
             */
            void detect(const cv::cuda::GpuMat &_frame, const cv::cuda::GpuMat &_mask);

            /**
             * Draws the detected features.
             *
             * @return The original image with the features drawn.
             */
            cv::Mat draw();
        };

        /**
         * Wrapper for the CUDA SURF feature detector.
         */
        class SURFFeatureDetector : public FeatureDetectorBase {
        private:
            /**
             * The CUDA SURF detector used to detect keypointsGPU and descriptorsGPU.
             */
            cv::Ptr<cv::cuda::SURF_CUDA> detector;

        public:

            /**
             * @constructor
             *
             * @ref opencv2/xfeatures2d/cuda.hpp -> cv::cuda::SURF_CUDA::create
             */
            explicit SURFFeatureDetector(double _hessianThreshold = 1000, int _nOctaves = 4,
                                         int _nOctaveLayers = 2, bool _extended = false, float _keypointsRatio = 0.01f,
                                         bool _upright = false);

            void specificDetect() override;

        };

        /**
         * Wrapper for the CUDA SURF feature detector.
         */
        class ORBFeatureDetector : public FeatureDetectorBase {
        private:

/**
             * The CUDA ORB detector used to detect keypoints and descriptors.
             */
            cv::Ptr<cv::cuda::ORB> detector;

        public:

            /**
             * @constructor
             *
             * @ref opencv2/cudafeatures2d.hpp -> cv::cuda::ORB::create
             */
            explicit ORBFeatureDetector(int nfeatures = 1e4,
                                        float scaleFactor = 1.2f,
                                        int nlevels = 8,
                                        int edgeThreshold = 31,
                                        int firstLevel = 0,
                                        int WTA_K = 2,
                                        int scoreType = cv::ORB::FAST_SCORE,
                                        int patchSize = 31,
                                        int fastThreshold = 20,
                                        bool blurForDescriptor = false);

            void specificDetect() override;

        };
    }
}

#endif //CAMERASTABILIZATION_FEATUREDETECTION_HPP
