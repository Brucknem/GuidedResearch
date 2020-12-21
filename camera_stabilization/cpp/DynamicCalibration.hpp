
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

                explicit DynamicCalibrator(int verbosity = 0) : verbosity(verbosity) {}

            public:
                void addTimestamp(const std::string &name, int minVerbosity) {
                    if (verbosity >= minVerbosity) {
                        timer.addTimestamp(name);
                    }
                }

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

                bool hasReferenceFrame() const {
                    return isReferenceFrameSet;
                }

                void detectKeyframe() {
                    mask.upload(fullMask_cpu);
                    detector->detectWithDescriptors(referenceFrame, mask, referenceKeypoints, referenceDescriptors,
                                                    false);
                    isReferenceFrameSet = true;
                }

                void setReferenceFrame(const cv::cuda::GpuMat &_referenceFrame) {
                    convertToGrayscale(_referenceFrame, referenceFrame);
                    detectKeyframe();
                }

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

                std::vector<cv::cuda::Stream> requestCudaStreams(int amount) {
                    while (cudaStreams.size() < amount) {
                        cudaStreams.emplace_back();
                    }
                    return cudaStreams;
                }

                void detect() {
                    convertToGrayscale(latestFrame, latestFrame);
                    detector->detectWithDescriptors(latestFrame, mask, latestKeypoints, latestDescriptors, false);
                    addTimestamp("detected descriptors", 2);
                }

                void match() {
                    detect();
                    cv::cuda::Stream stream = requestCudaStreams(1)[0];
                    matcher->knnMatchAsync(latestDescriptors, referenceDescriptors, knnMatches, 2, cv::noArray(),
                                           stream); // find matches
                    stream.waitForCompletion();
                    addTimestamp("matched knn features", 2);
                }

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

                void setMask() {
                    setMask(fullMask_cpu);
                }

                void setMask(const cv::Mat &_mask) {
                    mask.upload(_mask);
                }

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

                long getRuntime() {
                    return timer.getTotalMilliseconds();
                }
            };

            class SurfBFDynamicCalibrator : public DynamicCalibrator {
            public:
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
