#include "opencv2/imgproc/imgproc.hpp"

#include "DynamicStabilization.hpp"
#include <opencv2/cudaarithm.hpp>

namespace providentia {
    namespace stabilization {

#pragma region DynamicStabilizerBase

        void DynamicStabilizerBase::stabilize(const cv::cuda::GpuMat &_frame) {
            clear();
            updateKeyframe();

            frameFeatureDetector->detect(_frame, segmentor->getBackgroundMask(_frame.size()));
            if (referenceFeatureDetector->isEmpty()) {
                // TODO copy/clone frameFeatureDetector
                referenceFeatureDetector->detect(_frame, segmentor->getBackgroundMask(_frame.size()));
            }

            matcher->match(frameFeatureDetector, referenceFeatureDetector);
            warper->warp(_frame, matcher);
            segmentor->segment(getStabilizedFrame());
            addTimestamp("Stabilization finished", 0);
        }

        void DynamicStabilizerBase::updateKeyframe() {
            if (!shouldUpdateKeyframe || currentIteration++ < warmUp) {
                return;
            }

            int newCount = cv::cuda::countNonZero(this->segmentor->getBackgroundMask(getStabilizedFrame().size()));
            int oldCount = cv::cuda::countNonZero(this->referenceFeatureDetector->getCurrentMask());
            if (oldCount == this->getStabilizedFrame().size().width * this->getStabilizedFrame().size().height ||
                newCount > oldCount) {
                this->referenceFeatureDetector->detect(this->getStabilizedFrame(),
                                                       this->segmentor->getBackgroundMask(
                                                               getStabilizedFrame().size()).clone());
            }
        }

        const cv::Mat &DynamicStabilizerBase::getHomography() const {
            return warper->getHomography();
        }

        const cv::cuda::GpuMat &DynamicStabilizerBase::getStabilizedFrame() const {
            return warper->getWarpedFrame();
        }

        DynamicStabilizerBase::DynamicStabilizerBase() : providentia::utils::TimeMeasurable(
                "DynamicStabilizerBase", 1) {
            warper = std::make_shared<FrameWarper>();
            segmentor = std::make_shared<providentia::segmentation::MOG2BackgroundSegmentor>(cv::Size(1920, 1200) / 10);
        }

        cv::Mat DynamicStabilizerBase::draw() {
            cv::Mat result;
            cv::hconcat(std::vector<cv::Mat>{cv::Mat(frameFeatureDetector->getOriginalFrame()),
                                             cv::Mat(getStabilizedFrame())},
                        result);
            return result;
        }

        const cv::cuda::GpuMat &DynamicStabilizerBase::getReferenceframe() const {
            return referenceFeatureDetector->getOriginalFrame();
        }

        const cv::cuda::GpuMat &DynamicStabilizerBase::getReferenceframeMask() const {
            return referenceFeatureDetector->getCurrentMask();
        }

        const std::shared_ptr<providentia::features::FeatureDetectorBase> &
        DynamicStabilizerBase::getFrameFeatureDetector() const {
            return frameFeatureDetector;
        }

        const std::shared_ptr<providentia::features::FeatureMatcherBase> &
        DynamicStabilizerBase::getMatcher() const {
            return matcher;
        }

        const std::shared_ptr<providentia::segmentation::BackgroundSegmentorBase> &
        DynamicStabilizerBase::getSegmentor() const {
            return segmentor;
        }

        const std::shared_ptr<FrameWarper> &
        DynamicStabilizerBase::getWarper() const {
            return warper;
        }

        bool DynamicStabilizerBase::isShouldUpdateKeyframe() const {
            return shouldUpdateKeyframe;
        }

        void DynamicStabilizerBase::setShouldUpdateKeyframe(bool _shouldUpdateKeyframe) {
            DynamicStabilizerBase::shouldUpdateKeyframe = _shouldUpdateKeyframe;
        }

        void DynamicStabilizerBase::setShouldUseFundamentalMatrix(bool shouldUseFundamentalMatrix) {
            matcher->setShouldUseFundamentalMatrix(shouldUseFundamentalMatrix);
        }

        DynamicStabilizerBase::~DynamicStabilizerBase() = default;

#pragma endregion DynamicStabilizerBase

#pragma region SURFBFDynamicStabilizer

        SURFBFDynamicStabilizer::SURFBFDynamicStabilizer(double _hessianThreshold,
                                                         int _nOctaves,
                                                         int _nOctaveLayers,
                                                         bool _extended,
                                                         float _keypointsRatio,
                                                         bool _upright) {
            auto detector = providentia::features::SURFFeatureDetector(_hessianThreshold, _nOctaves,
                                                                       _nOctaveLayers, _extended,
                                                                       _keypointsRatio, _upright);
            frameFeatureDetector = std::make_shared<providentia::features::SURFFeatureDetector>(detector);
            referenceFeatureDetector = std::make_shared<providentia::features::SURFFeatureDetector>(detector);
            matcher = std::make_shared<providentia::features::BruteForceFeatureMatcher>(cv::NORM_L2);
            setName(typeid(*this).name());
        }

        SURFBFDynamicStabilizer::~SURFBFDynamicStabilizer() = default;

#pragma endregion SURFBFDynamicStabilizer

#pragma region ORBBFDynamicStabilizer

        ORBBFDynamicStabilizer::ORBBFDynamicStabilizer(int nfeatures, float scaleFactor,
                                                       int nlevels, int edgeThreshold,
                                                       int firstLevel, int WTA_K,
                                                       int scoreType,
                                                       int patchSize, int fastThreshold,
                                                       bool blurForDescriptor) {
            auto detector = providentia::features::ORBFeatureDetector(nfeatures, scaleFactor,
                                                                      nlevels, edgeThreshold,
                                                                      firstLevel, WTA_K, scoreType,
                                                                      patchSize, fastThreshold,
                                                                      blurForDescriptor);
            frameFeatureDetector = std::make_shared<providentia::features::ORBFeatureDetector>(detector);
            referenceFeatureDetector = std::make_shared<providentia::features::ORBFeatureDetector>(detector);
            matcher = std::make_shared<providentia::features::BruteForceFeatureMatcher>(cv::NORM_HAMMING);
            setName(typeid(*this).name());
        }

        ORBBFDynamicStabilizer::~ORBBFDynamicStabilizer() = default;

#pragma endregion ORBBFDynamicStabilizer

        FastFREAKBFDynamicStabilizer::FastFREAKBFDynamicStabilizer(int threshold,
                                                                   bool nonmaxSuppression,
                                                                   cv::FastFeatureDetector::DetectorType type,
                                                                   int max_npoints,
                                                                   bool orientationNormalized,
                                                                   bool scaleNormalized,
                                                                   float patternScale,
                                                                   int nOctaves,
                                                                   const std::vector<int> &selectedPairs) {
            auto detector = providentia::features::FastFREAKFeatureDetector(threshold,
                                                                            nonmaxSuppression, type,
                                                                            max_npoints,
                                                                            orientationNormalized,
                                                                            scaleNormalized,
                                                                            patternScale, nOctaves,
                                                                            selectedPairs);
            frameFeatureDetector = std::make_shared<providentia::features::FastFREAKFeatureDetector>(detector);
            referenceFeatureDetector = std::make_shared<providentia::features::FastFREAKFeatureDetector>(detector);
            matcher = std::make_shared<providentia::features::BruteForceFeatureMatcher>(cv::NORM_HAMMING);
            setName(typeid(*this).name());
        }

        FastFREAKBFDynamicStabilizer::~FastFREAKBFDynamicStabilizer() = default;

    }
}
