
#ifndef DYNAMICSTABILIZATION_DYNAMICCALIBRATION_HPP
#define DYNAMICSTABILIZATION_DYNAMICCALIBRATION_HPP


#include <cstdio>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
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

namespace providentia {
    namespace calibration {
        namespace dynamic {

            /**
             * Base class for the dynamic calibration algorithms.
             */
            class DynamicCalibrator {
            protected:
                bool isReferenceFrameSet = false;
                cv::cuda::GpuMat latestFrame, latestColorFrame, referenceFrame, stabilizedFrame;
                cv::cuda::GpuMat latestKeypoints, referenceKeypoints;
                std::vector<cv::KeyPoint> latestKeypoints_cpu, referenceKeypoints_cpu;
                cv::cuda::GpuMat latestDescriptors, referenceDescriptors;
                cv::cuda::GpuMat mask;
                cv::Ptr<cv::cuda::DescriptorMatcher> matcher;
                cv::Ptr<cv::cuda::SURF_CUDA> detector;
                std::vector<cv::cuda::Stream> cudaStreams;
                cv::cuda::GpuMat knnMatches;
                std::vector<std::vector<cv::DMatch>> knnMatches_cpu;
                const float ratio_thresh = 0.75f;
                std::vector<cv::DMatch> goodMatches;
                std::vector<cv::Point2f> latestMatchedPoints;
                std::vector<cv::Point2f> referenceMatchedPoints;
                cv::Mat homography;
                cv::Mat stabilizedFrame_cpu;
                const cv::Mat fullMask_cpu = cv::Mat(referenceFrame.size(), CV_8UC1, cv::Scalar(1));
                providentia::utils::TimeMeasurable timer;
                int verbosity;

                /**
                 * Constructor.
                 *
                 * @param verbosity The verbosity of the time measuring. <br>
                 *          0: No measuring. <br>
                 *          1: Measuring only start to end. <br>
                 *          2: Measuring every step of the algorithm. <br>
                 */
                explicit DynamicCalibrator(int verbosity = 0) : verbosity(verbosity) {}

                /**
                 * Detects the keypoints and descriptors in the reference frame.
                 */
                void detectKeyframe() {
                    setMask();
                    detector->detectWithDescriptors(referenceFrame, mask, referenceKeypoints, referenceDescriptors,
                                                    false);
                    isReferenceFrameSet = true;
                }

            public:

                /**
                 * Adds a timestamp to the time measuring instance. Only adds a timestamp if the verbosity requirement is met.
                 *
                 * @param name The name of the last algorithmic step.
                 * @param minVerbosity The minimum verbosity needed to add a timestamp.
                 */
                void addTimestamp(const std::string &name, int minVerbosity) {
                    if (verbosity >= minVerbosity) {
                        timer.addTimestamp(name);
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
                    if (verbosity > 2) {}
                    addTimestamp("converted frame to grayscale", 2);
                }

                /**
                 * Has the reference frame been set.
                 *
                 * @return true if set, false else.
                 */
                bool hasReferenceFrame() const {
                    return isReferenceFrameSet;
                }

                /**
                 * Sets the reference frame.
                 *
                 * @param _referenceFrame A CPU frame.
                 */
                void setReferenceFrame(const cv::cuda::GpuMat &_referenceFrame) {
                    convertToGrayscale(_referenceFrame, referenceFrame);
                    detectKeyframe();
                }

                /**
                 * Sets the reference frame.
                 *
                 * @param _referenceFrame A GPU frame.
                 */
                void setReferenceFrame(const cv::Mat &_referenceFrame) {
                    cv::Mat grayscale;
                    if (_referenceFrame.channels() == 3) {
                        cv::cvtColor(_referenceFrame, grayscale, cv::COLOR_BGR2GRAY);
                    } else if (_referenceFrame.channels() == 1) {
                        grayscale = _referenceFrame.clone();
                    } else {
                        throw std::invalid_argument("Given cv::Mat is has neither 1 or 3 channels.");
                    }
                    referenceFrame.upload(grayscale);
                    detectKeyframe();
                }

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
                 * Detects keypoints and descriptors in the latest frame.
                 */
                void detect() {
                    convertToGrayscale(latestFrame, latestFrame);
                    detector->detectWithDescriptors(latestFrame, mask, latestKeypoints, latestDescriptors, false);
                    addTimestamp("detected descriptors", 2);
                }

                /**
                 * Matches the descriptors of the latest and reference frame.
                 */
                void match() {
                    detect();
                    cv::cuda::Stream stream = requestCudaStreams(1)[0];
                    matcher->knnMatchAsync(latestDescriptors, referenceDescriptors, knnMatches, 2, cv::noArray(),
                                           stream); // find matches
                    stream.waitForCompletion();
                    addTimestamp("matched knn features", 2);
                }

                /**
                 * Finds the homography between the latest and reference frame.
                 */
                void findHomography() {
                    match();
                    matcher->knnMatchConvert(knnMatches, knnMatches_cpu);
                    addTimestamp("moved matches to cpu", 2);

                    //-- Filter matches using the Lowe's ratio test
                    goodMatches.clear();
                    for (auto &knnMatch_cpu : knnMatches_cpu) {
                        if (knnMatch_cpu[0].distance < ratio_thresh * knnMatch_cpu[1].distance) {
                            goodMatches.push_back(knnMatch_cpu[0]);
                        }
                    }
                    addTimestamp("filtered matches", 2);

                    //-- Draw matches
                    // cv::Mat img_matches;
                    // drawMatches(frame, cpuKeys1, cv::Mat(referenceFrame), cpuKeys2, goodMatches, img_matches, cv::Scalar::all(-1),
                    //             cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                    // std::vector<cv::Mat> results{img_matches};
//            std::vector<cv::Mat> results{frame};

                    detector->downloadKeypoints(latestKeypoints, latestKeypoints_cpu);
                    detector->downloadKeypoints(referenceKeypoints, referenceKeypoints_cpu);
                    addTimestamp("downloaded keypoints", 2);

                    //-- Localize the object
                    latestMatchedPoints.clear();
                    referenceMatchedPoints.clear();
                    for (auto &goodMatch : goodMatches) {
                        //-- Get the keypoints from the good matches
                        latestMatchedPoints.push_back(latestKeypoints_cpu[goodMatch.queryIdx].pt);
                        referenceMatchedPoints.push_back(referenceKeypoints_cpu[goodMatch.trainIdx].pt);
                    }
                    addTimestamp("extracted matched points", 2);

                    homography = cv::findHomography(latestMatchedPoints, referenceMatchedPoints, cv::RANSAC);
                    addTimestamp("found homography", 2);
                }

                /**
                 * Sets the detection mask to a mask keeping all pixels.
                 */
                void setMask() {
                    setMask(fullMask_cpu);
                }

                /**
                 * Sets the given mask.
                 *
                 * @param _mask A binary mask indicating the pixels to exclude during detection.
                 */
                void setMask(const cv::Mat &_mask) {
                    mask.upload(_mask);
                }

                /**
                 * Stabilizes the given frame using a reference frame.
                 *
                 * @param _frame The frame to stabilize.
                 * @return The stabilized frame.
                 */
                cv::Mat stabilize(const cv::Mat &_frame) {
                    timer.clear();
                    latestFrame.upload(_frame);
                    latestColorFrame.upload(_frame);
                    addTimestamp("current frame uploaded", 2);
                    findHomography();
                    cv::cuda::warpPerspective(latestColorFrame, stabilizedFrame, homography, latestColorFrame.size(),
                                              cv::INTER_LINEAR);
                    addTimestamp("warped frame", 2);

                    stabilizedFrame.download(stabilizedFrame_cpu);
                    addTimestamp("downloaded stabilized frame", 1);
                    return stabilizedFrame_cpu;
                }

                /**
                 * Gets the total runtime of the stabilization algorithm.
                 * @return
                 */
                long getRuntime() {
                    return timer.getTotalMilliseconds();
                }
            };

            /**
             * Dynamic calibration using SURF features and a Brute Force matching algorithm.
             */
            class SurfBFDynamicCalibrator : public DynamicCalibrator {
            public:
                /**
                 * Constructor.
                 *
                 * @param hessian Threshold for hessian keypoint detector used in SURF.
                 * @param norm The norm used in the matcher. One of cv::NORM_L1, cv::NORM_L2.
                 * @param verbosity The verbosity of the time measuring.
                 * @param _nOctaves Number of pyramid octaves the keypoint detector will use.
                 * @param _nOctaveLayers Number of octave layers within each octave.
                 * @param _extended Extended descriptor flag (true - use extended 128-element descriptors; false - use
                 *                  64-element descriptors).
                 * @param _keypointsRatio
                 * @param _upright Up-right or rotated features flag (true - do not compute orientation of features;
                 *                  false - compute orientation).
                 */
                explicit SurfBFDynamicCalibrator(int hessian = 1000, int norm = cv::NORM_L2, int verbosity = 0,
                                                 int _nOctaves = 4,
                                                 int _nOctaveLayers = 2, bool _extended = false,
                                                 float _keypointsRatio = 0.01f,
                                                 bool _upright = false) : DynamicCalibrator(verbosity) {
                    timer = providentia::utils::TimeMeasurable("Surf BF");
                    detector = cv::cuda::SURF_CUDA::create(hessian, _nOctaves, _nOctaveLayers, _extended,
                                                           _keypointsRatio, _upright);
                    matcher = cv::cuda::DescriptorMatcher::createBFMatcher(norm);
                }
            };

        }
    }
} // namespace providentia::calibration::dynamic
#endif //DYNAMICSTABILIZATION_DYNAMICCALIBRATION_HPP
