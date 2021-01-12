
#ifndef DYNAMICSTABILIZATION_DYNAMICCALIBRATION_HPP
#define DYNAMICSTABILIZATION_DYNAMICCALIBRATION_HPP


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
#include "Utils.hpp"
#include "BackgroundSegmentation.h"

namespace providentia {
    namespace calibration {
        namespace dynamic {

            /**
             * Base class for the dynamic calibration algorithms.
             */
            class DynamicCalibrator : public virtual providentia::utils::TimeMeasurable {
            protected:
                cv::cuda::GpuMat latestFrame, latestColorFrame, referenceFrame, stabilizedFrame, stabilizedFrameScaled;
                cv::cuda::GpuMat latestKeypoints, referenceKeypoints;
                std::vector<cv::KeyPoint> latestKeypoints_cpu, referenceKeypoints_cpu;
                cv::cuda::GpuMat latestDescriptors, referenceDescriptors;
                cv::cuda::GpuMat latestMask, referenceMask;
                cv::Ptr<cv::cuda::DescriptorMatcher> matcher;
                cv::Ptr<cv::cuda::SURF_CUDA> detector;
                std::vector<cv::cuda::Stream> cudaStreams;
                cv::cuda::GpuMat knnMatches;
                std::vector<std::vector<cv::DMatch>> knnMatches_cpu;
                float ratio_thresh = 0.75f;
                std::vector<cv::DMatch> goodMatches;
                std::vector<cv::Point2f> latestMatchedPoints;
                std::vector<cv::Point2f> referenceMatchedPoints;
                cv::Mat homography;

                cv::Size size;

                bool withBackground = false;
                segmentation::MOG2 backgroundSegmentation;

                /**
                 * Detects the keypointsGPU and descriptorsGPU in the reference frame.
                 */
                void detectKeyframe() {
                    setReferenceMask();
                    // addTimestamp("reference mask set", 3);
                    detector->detectWithDescriptors(referenceFrame, referenceMask, referenceKeypoints,
                                                    referenceDescriptors,
                                                    false);
                    // addTimestamp("reference frame detected", 3);
                }

                virtual void setReferenceMask() {
                    if (referenceMask.empty()) {
                        referenceMask.upload(cv::Mat::ones(referenceFrame.size(), CV_8UC1) * 255);
                    }
                }

                /**
                     * Constructor.
                     */
                explicit DynamicCalibrator() {
                    setNameAndVerbosity("Dynamic Calibrator", 0);
                }

                /**
                     * Constructor.
                     */
                explicit DynamicCalibrator(bool _withBackground) : DynamicCalibrator() {
                    withBackground = _withBackground;
                }

            public:

                /**
                 * Sets the reference frame.
                 *
                 * @param _referenceFrame A CPU frame.
                 */
                void setReferenceFrame(const cv::cuda::GpuMat &_referenceFrame) {
                    convertToGrayscale(_referenceFrame, referenceFrame);
                    cv::cuda::resize(referenceFrame, referenceFrame, size);
                    // addTimestamp("reference frame set", 3);
                    detectKeyframe();
                }


                virtual void setScaleFactor(int width, int height) {
                    setScaleFactor(cv::Size(width, height));
                }

                virtual void setScaleFactor(cv::Size _size) {
                    if (_size.height == size.height && _size.width == size.width) {
                        return;
                    }
                    size = std::move(_size);
                    if (!latestFrame.empty()) {
                        cv::cuda::resize(latestFrame, latestFrame, size);
                        cv::cuda::resize(latestColorFrame, latestColorFrame, size);
                    }
                    if (!latestMask.empty()) {
                        cv::cuda::resize(latestMask, latestMask, size);
                    }
                    if (!referenceFrame.empty()) {
                        cv::cuda::resize(referenceFrame, referenceFrame, size);
                    }
                    if (!referenceMask.empty()) {
                        cv::cuda::resize(referenceMask, referenceMask, size);
                    }
                }

                cv::Mat getHomography() {
                    return homography;
                }

                cv::cuda::GpuMat getReferenceFrame() {
                    return referenceFrame;
                }

                cv::cuda::GpuMat getReferenceMask() {
                    return referenceMask;
                }

                cv::cuda::GpuMat getLatestFrame() {
                    return latestFrame;
                }

                cv::cuda::GpuMat getLatestMask() {
                    return latestMask;
                }

                cv::cuda::GpuMat getStabilizedFrame() {
                    return stabilizedFrame;
                }

                /**
                 * Sets the detection latestMask to a latestMask keeping all pixels.
                 */
                void setMask() {
                    setMask(cv::Mat::ones(latestFrame.size(), CV_8UC1) * 255);
                }

                /**
                 * Sets the given latestMask.
                 *
                 * @param _mask A binary latestMask indicating the pixels to exclude during detection.
                 */
                void setMask(const cv::Mat &_mask) {
                    latestMask.upload(_mask);
                }

                /**
                 * Sets the given latestMask.
                 *
                 * @param _mask A binary latestMask indicating the pixels to exclude during detection.
                 */
                void setMask(const cv::cuda::GpuMat &_mask) {
                    latestMask = _mask;
                }

                virtual void resize(const cv::cuda::GpuMat &_frame) {
                    if (size.empty()) {
                        size = _frame.size();
                    }
                    if (latestFrame.empty() || latestFrame.size() != _frame.size()) {
                        cv::cuda::resize(_frame, latestFrame, size);
                    } else {
                        latestFrame = _frame;
                    }
                }

                void setLatestFrame(const cv::cuda::GpuMat &_frame) {
                    resize(_frame);
                    latestColorFrame = latestFrame.clone();
                    if (!backgroundSegmentation.empty()) {
                        cv::cuda::resize(backgroundSegmentation.getBackgroundMaskGpu(), latestMask, latestFrame.size());
                    }
                    if (latestMask.empty()) {
                        setMask();
                    }
                }

                /**
                 * Converts the given frame to grayscale.
                 *
                 * @param input The frame to convert.
                 * @param output The frame to write the grayscale frame to.
                 */
                void convertToGrayscale(const cv::cuda::GpuMat &input, cv::cuda::GpuMat &output) {
                    if (input.channels() == 3) {
                        cv::cuda::cvtColor(input, output, cv::COLOR_BGR2GRAY);
                    } else if (input.channels() == 1) {
                        output = input;
                    } else {
                        throw std::invalid_argument("Given cv::Mat is has neighter 1 or 3 channels.");
                    }
                    // addTimestamp("converted frame to grayscale", 3);
                }

                /**
                 * Has the reference frame been set.
                 *
                 * @return true if set, false else.
                 */
                bool hasReferenceFrame() const {
                    return !referenceFrame.empty();
                }


//                /**
//                 * Sets the reference frame.
//                 *
//                 * @param _referenceFrame A GPU frame.
//                 */
//                virtual void setReferenceFrame(const cv::cuda::GpuMat &_referenceFrame) {
//                    cv::Mat grayscale;
//                    if (_referenceFrame.channels() == 3) {
//                        cv::cvtColor(_referenceFrame, grayscale, cv::COLOR_BGR2GRAY);
//                    } else if (_referenceFrame.channels() == 1) {
//                        grayscale = _referenceFrame.clone();
//                    } else {
//                        throw std::invalid_argument("Given cv::Mat is has neither 1 or 3 channels.");
//                    }
//                    referenceFrame.upload(grayscale);
//                    detectKeyframe();
//                }

                /**
                 * Appends streams to the buffer so that there are at least the requested amount of streams available.
                 *
                 * @param amount The number of needed cuda streams.
                 * @return The cuda streams.
                 */
                std::vector<cv::cuda::Stream> requestCudaStreams(int amount) {
                    while (cudaStreams.size() < amount) {
                        cudaStreams.emplace_back();
                    }
                    return cudaStreams;
                }

                /**
                 * Detects keypointsGPU and descriptorsGPU in the latest frame.
                 */
                void detect() {
                    convertToGrayscale(latestFrame, latestFrame);
                    detector->detectWithDescriptors(latestFrame, latestMask, latestKeypoints, latestDescriptors, false);
                    // addTimestamp("detected descriptorsGPU", 3);
                }

                /**
                 * Matches the descriptorsGPU of the latest and reference frame.
                 */
                void match() {
                    detect();
                    cv::cuda::Stream stream = requestCudaStreams(1)[0];
                    matcher->knnMatchAsync(latestDescriptors, referenceDescriptors, knnMatches, 2, cv::noArray(),
                                           stream); // find matches
                    // addTimestamp("matched knn features", 3);
                    matcher->knnMatchConvert(knnMatches, knnMatches_cpu);
                    waitForCompletion();
                    // addTimestamp("moved matches to cpu", 3);
                }

                void waitForCompletion() {
                    for (cv::cuda::Stream &stream : cudaStreams) {
                        stream.waitForCompletion();
                    }
                }

                /**
                 * Finds the homography between the latest and reference frame.
                 */
                void findHomography() {
                    match();

                    //-- Filter matches using the Lowe's ratio test
                    goodMatches.clear();
                    for (auto &knnMatch_cpu : knnMatches_cpu) {
                        if (knnMatch_cpu[0].distance < ratio_thresh * knnMatch_cpu[1].distance) {
                            goodMatches.push_back(knnMatch_cpu[0]);
                        }
                    }
                    // addTimestamp("filtered matches", 3);

                    detector->downloadKeypoints(latestKeypoints, latestKeypoints_cpu);
                    detector->downloadKeypoints(referenceKeypoints, referenceKeypoints_cpu);
                    // addTimestamp("downloaded keypointsGPU", 3);

                    //-- Localize the object
                    latestMatchedPoints.clear();
                    referenceMatchedPoints.clear();
                    for (auto &goodMatch : goodMatches) {
                        //-- Get the keypointsGPU from the good matches
                        latestMatchedPoints.push_back(latestKeypoints_cpu[goodMatch.queryIdx].pt);
                        referenceMatchedPoints.push_back(referenceKeypoints_cpu[goodMatch.trainIdx].pt);
                    }
                    // addTimestamp("extracted matched points", 3);

                    if (goodMatches.size() < 4) {
                        homography = cv::Mat::eye(3, 3, CV_8UC1);
                    } else {
                        homography = cv::findHomography(latestMatchedPoints, referenceMatchedPoints, cv::RANSAC);
                    }
                    // addTimestamp("found homography", 3);
                }

                cv::Mat draw(double renderingScaleFactor = 1.0) {
                    cv::Mat drawFrame;
                    if (getReferenceFrame().empty()) {
                        drawFrame = cv::Mat(latestFrame);
                    } else {
                        drawMatches(cv::Mat(getLatestFrame()), latestKeypoints_cpu, cv::Mat(getReferenceFrame()),
                                    referenceKeypoints_cpu,
                                    goodMatches, drawFrame, cv::Scalar::all(-1),
                                    cv::Scalar::all(-1), std::vector<char>(),
                                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                    }
                    cv::resize(drawFrame, drawFrame, cv::Size(), renderingScaleFactor, renderingScaleFactor);
                    cv::imshow("test", drawFrame);
                    while ((char) cv::waitKey(1) == 27) {
                        break;
                    }
                    return drawFrame;
                }

                /**
                 * Stabilizes the given frame using a reference frame.
                 *
                 * @param _frame The frame to stabilize.
                 * @return The stabilized frame.
                 */
                virtual void stabilize(const cv::cuda::GpuMat &_frame) {
                    stabilize(_frame, cv::INTER_LINEAR);
                }

                virtual void stabilize(const cv::cuda::GpuMat &_frame,
                                       cv::InterpolationFlags findHomographyInterpolationAlgorithm) {
                    clear();

                    int referenceMaskCount = 0;
                    if (!referenceMask.empty()) {
                        referenceMaskCount = cv::cuda::countNonZero(referenceMask);
                    }
                    setLatestFrame(_frame);

                    int latestMaskCount = 0;
                    if (!latestMask.empty()) {
                        latestMaskCount = cv::cuda::countNonZero(latestMask);
                    }

                    if (withBackground && (!stabilizedFrame.empty() &&
                                           (referenceMaskCount == 0 ||
                                            referenceMaskCount == latestFrame.rows * latestFrame.cols ||
                                            latestMaskCount > referenceMaskCount))) {
                        referenceMask = latestMask.clone();
                        setReferenceFrame(stabilizedFrame);
                    }

                    if (!hasReferenceFrame()) {
                        setReferenceFrame(latestFrame);
                        clear();
                        if (!hasReferenceFrame()) {
                            stabilizedFrame = latestFrame;
                            return;
                        }
                    }

                    // addTimestamp("current frame set", 3);
                    findHomography();
                    cv::cuda::warpPerspective(latestColorFrame, stabilizedFrame, homography, latestColorFrame.size(),
                                              cv::INTER_LINEAR);
                    if (withBackground) {
                        cv::Size preSize = stabilizedFrame.size();
                        double scaleFactor = 1.0;
                        cv::cuda::resize(stabilizedFrame, stabilizedFrameScaled,
                                         cv::Size(preSize.width * scaleFactor, preSize.height * scaleFactor));
                        backgroundSegmentation.apply(stabilizedFrameScaled);
                    }
                    addTimestamp("warped frame", 0);
                }
            };

            class ExtendedDynamicCalibrator : public virtual DynamicCalibrator {
            private:
                std::shared_ptr<DynamicCalibrator> initialGuessCalibrator;
                double scaleFactor = 0.5;

                providentia::segmentation::MOG2 initialGuessBackgroundSegmentation;
                cv::cuda::GpuMat initialBackgroundMask;
                cv::cuda::GpuMat initialStabilized;
                cv::cuda::GpuMat initialWarpedFrame;

                int warmup = 50;
                int frameNumber = 0;

            protected:
                explicit ExtendedDynamicCalibrator(const DynamicCalibrator &_initialGuessCalibrator,
                                                   bool withBackground = false) : DynamicCalibrator(withBackground) {
                    initialGuessCalibrator = std::make_shared<DynamicCalibrator>(_initialGuessCalibrator);
                    setNameAndVerbosity("Extended Dynamic Calibrator", 0);
                }

            public:
                void setScaleFactor(int width, int height) override {
                    setScaleFactor(cv::Size(width, height));
                }

                void setScaleFactor(cv::Size _size) override {
                    DynamicCalibrator::setScaleFactor(_size);
                    initialGuessCalibrator->setScaleFactor(size.width * scaleFactor, size.height * scaleFactor);
                }

                void stabilize(const cv::cuda::GpuMat &_frame) override {
                    clear();

                    resize(_frame);
                    initialGuessCalibrator->setScaleFactor(size.width * scaleFactor, size.height * scaleFactor);

                    // Downsample
//                    cv::cuda::resize(latestFrame, resizeBuffer, latestFrame.size(), scaleFactor, scaleFactor);
                    // addTimestamp("warped frame", 2);

                    // Stabilize downsampled for initial homography guess
                    initialGuessCalibrator->stabilize(_frame);
                    initialStabilized = initialGuessCalibrator->getStabilizedFrame();
                    // addTimestamp("Initial stabilization", 1);

                    // Use initial guess for background segmentation
                    initialGuessBackgroundSegmentation.apply(initialStabilized);
                    // addTimestamp("Background segmentation", 1);

                    // Resize segmentation to original size
                    initialBackgroundMask.upload(initialGuessBackgroundSegmentation.getBackgroundMask());
                    cv::cuda::resize(initialBackgroundMask, initialBackgroundMask, latestFrame.size());
                    // addTimestamp("Resize Background segmentation", 2);

                    // Set segmentation mask
                    setMask(initialBackgroundMask);
                    // addTimestamp("Set Background mask", 2);

                    // Warp original frame using the initial guess homography
                    cv::cuda::warpPerspective(latestFrame, initialWarpedFrame, initialGuessCalibrator->getHomography(),
                                              latestFrame.size(),
                                              cv::INTER_LINEAR);
                    // addTimestamp("Warp original frame", 2);

                    // Update reference frame
                    if (!hasReferenceFrame()) {
                        setReferenceFrame(initialWarpedFrame);
                        // addTimestamp("Set reference frame", 2);
                    }

                    // Stabilize
                    DynamicCalibrator::stabilize(initialWarpedFrame);
                    // addTimestamp("Stabilization", 2);

                    // Update reference frame
                    if (frameNumber++ > warmup) {
                        int referenceMaskNonZero = cv::cuda::countNonZero(getReferenceMask());
                        int initialMaskNonZero = cv::cuda::countNonZero(initialBackgroundMask);
                        // addTimestamp("Count nonzero", 2);

                        if (referenceMaskNonZero == 0 || initialMaskNonZero > referenceMaskNonZero) {
                            setReferenceFrame(getStabilizedFrame());
                            // addTimestamp("Set reference frame", 2);
                        }
                    }
                    addTimestamp("Finished", 0);
                }

                void setReferenceMask() override {
                    referenceMask = initialBackgroundMask;
                }
            };


            /**
             * Dynamic calibration using SURF features and a Brute Force matching algorithm.
             */
            class SurfBFDynamicCalibrator : public virtual DynamicCalibrator {
            public:
                /**
                 * Constructor.
                 *
                 * @param hessian Threshold for hessian keypoint detector used in SURF.
                 * @param norm The norm used in the matcher. One of cv::NORM_L1, cv::NORM_L2.
                 * @param _nOctaves Number of pyramid octaves the keypoint detector will use.
                 * @param _nOctaveLayers Number of octave layers within each octave.
                 * @param _extended Extended descriptor flag (true - use extended 128-element descriptorsGPU; false - use
                 *                  64-element descriptorsGPU).
                 * @param _keypointsRatio
                 * @param _upright Up-right or rotated features flag (true - do not compute orientation of features;
                 *                  false - compute orientation).
                 */
                explicit SurfBFDynamicCalibrator(int hessian = 1000, int norm = cv::NORM_L2,
                                                 bool withBackground = false,
                                                 int _nOctaves = 4,
                                                 int _nOctaveLayers = 2, bool _extended = false,
                                                 float _keypointsRatio = 0.01f,
                                                 bool _upright = false) : DynamicCalibrator(withBackground) {
                    detector = cv::cuda::SURF_CUDA::create(hessian, _nOctaves, _nOctaveLayers, _extended,
                                                           _keypointsRatio, _upright);
                    matcher = cv::cuda::DescriptorMatcher::createBFMatcher(norm);
                    setNameAndVerbosity("SURF BF", 0);
                }
            };

            class ExtendedSurfBFDynamicCalibrator
                    : public SurfBFDynamicCalibrator, public ExtendedDynamicCalibrator {
            public:
                /**
                 * Constructor.
                 *
                 * @param hessian Threshold for hessian keypoint detector used in SURF.
                 * @param norm The norm used in the matcher. One of cv::NORM_L1, cv::NORM_L2.
                 * @param verbosity The verbosity of the time measuring.
                 * @param _nOctaves Number of pyramid octaves the keypoint detector will use.
                 * @param _nOctaveLayers Number of octave layers within each octave.
                 * @param _extended Extended descriptor flag (true - use extended 128-element descriptorsGPU; false - use
                 *                  64-element descriptorsGPU).
                 * @param _keypointsRatio
                 * @param _upright Up-right or rotated features flag (true - do not compute orientation of features;
                 *                  false - compute orientation).
                 */
                explicit ExtendedSurfBFDynamicCalibrator(int hessian = 1000,
                                                         int norm = cv::NORM_L2,
                                                         int _nOctaves = 4,
                                                         int _nOctaveLayers = 2, bool _extended = false,
                                                         float _keypointsRatio = 0.01f,
                                                         bool _upright = false) : ExtendedDynamicCalibrator(
                        SurfBFDynamicCalibrator()), SurfBFDynamicCalibrator(
                        hessian, norm, _nOctaves, _nOctaveLayers, _extended, _keypointsRatio,
                        _upright) {
                    setNameAndVerbosity("Extended SURF BF", 0);
                }

            };


            class TwoPassDynamicCalibrator : public DynamicCalibrator {
            private:
                int frameNumber = 0;
                int warmUp = 50;

                void updateReferenceFrame() {
                    if (frameNumber++ < warmUp) {
                        return;
                    }
                    cv::Mat newMask = latestBackgroundSegmentation.getBackgroundMask();
                    cv::cuda::GpuMat oldMask = getReferenceMask();
                    int newMaskNonZero = cv::countNonZero(newMask);
                    int oldMaskNonZero = cv::countNonZero(oldMask);
                    if (oldMaskNonZero == newMask.rows * newMask.cols || newMaskNonZero > oldMaskNonZero) {
                        setReferenceFrame(stabilizedFrame);
                    }
                }

                void setReferenceMask() override {
                    referenceMask.upload(latestBackgroundSegmentation.getBackgroundMask());
                }

            protected:
                explicit TwoPassDynamicCalibrator() {
                    setNameAndVerbosity("Two Pass Dynamic Calibrator", 0);
                }

            public:
                providentia::segmentation::MOG2 latestBackgroundSegmentation;

                void stabilize(const cv::cuda::GpuMat &_frame) override {
                    clear();
                    if (latestBackgroundSegmentation.getForegroundMask().empty()) {
                        latestBackgroundSegmentation.apply(_frame);
                    }
                    DynamicCalibrator::stabilize(_frame);
                    // addTimestamp("First pass", 2);
                    latestBackgroundSegmentation.apply(stabilizedFrame);
                    // addTimestamp("Segmentation", 2);
                    updateReferenceFrame();
                    // addTimestamp("Update Reference Frame", 2);
                    setMask(latestBackgroundSegmentation.getBackgroundMask());
                    // addTimestamp("Update Frame Mask", 2);

//                    // addTimestamp("Updated reference frame", 3);
                    DynamicCalibrator::stabilize(stabilizedFrame);
                    // addTimestamp("Second pass", 2);
                }
            };
        }
    }
} // namespace providentia::calibration::dynamic
#endif //DYNAMICSTABILIZATION_DYNAMICCALIBRATION_HPP
