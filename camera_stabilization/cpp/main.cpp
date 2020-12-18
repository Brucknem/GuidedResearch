#include <stdio.h>
#include <iostream>
#include "lib/CameraStabilization/CameraStabilization.hpp"
#include "lib/ImageUtils/ImageUtils.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include <opencv2/cudafeatures2d.hpp>
#include "opencv2/calib3d.hpp"

void detectORBKeypoints(cv::Mat frame, cv::Ptr<cv::cuda::Feature2DAsync> detector, cv::cuda::Stream cudaStream, cv::cuda::GpuMat &keypoints, cv::cuda::GpuMat &descriptors)
{
    cv::cuda::GpuMat frame_gpu;
    frame_gpu.upload(frame);
    detector->detectAndComputeAsync(frame_gpu, cv::noArray(), keypoints, descriptors, false, cudaStream);
}

std::vector<cv::Mat> orb_bf(cv::Mat frame, cv::Mat referenceFrame, cv::Ptr<cv::cuda::Feature2DAsync> detector, cv::Ptr<cv::cuda::DescriptorMatcher> matcher)
{
    cv::cuda::Stream cudaStream1, cudaStream2;
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::cvtColor(referenceFrame, referenceFrame, cv::COLOR_BGR2GRAY);

    cv::cuda::GpuMat keys1, keys2;                // this holds the keys detected
    cv::cuda::GpuMat desc1, desc2;                // this holds the descriptors for the detected keypoints
    std::vector<cv::KeyPoint> cpuKeys1, cpuKeys2; // holds keypoints downloaded from gpu
    detectORBKeypoints(frame, detector, cudaStream1, keys1, desc1);
    cudaStream1.waitForCompletion();
    detector->convert(keys1, cpuKeys1); // download keys to cpu if needed for anything...like displaying or whatever
    detectORBKeypoints(referenceFrame, detector, cudaStream2, keys2, desc2);
    cudaStream2.waitForCompletion();
    detector->convert(keys2, cpuKeys2); // download keys to cpu if needed for anything...like displaying or whatever


    std::vector<std::vector<cv::DMatch>> cpuKnnMatches;
    cv::cuda::GpuMat gpuKnnMatches;                                                     // holds matches on gpu
    matcher->knnMatchAsync(desc1, desc2, gpuKnnMatches, 2, cv::noArray(), cudaStream1); // find matches
    cudaStream1.waitForCompletion();

    matcher->knnMatchConvert(gpuKnnMatches, cpuKnnMatches); // download matches from gpu and put into vector<vector<DMatch>> form on cpu

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < cpuKnnMatches.size(); i++)
    {
        if (cpuKnnMatches[i][0].distance < ratio_thresh * cpuKnnMatches[i][1].distance)
        {
            good_matches.push_back(cpuKnnMatches[i][0]);
        }
    }


    //-- Draw matches
    cv::Mat img_matches;
    drawMatches(frame, cpuKeys1, referenceFrame, cpuKeys2, good_matches, img_matches, cv::Scalar::all(-1),
                cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    std::vector<cv::Mat> results{img_matches};

    //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
    for (size_t i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back(cpuKeys1[good_matches[i].queryIdx].pt);
        scene.push_back(cpuKeys2[good_matches[i].trainIdx].pt);
    }
    cv::Mat H = findHomography(obj, scene, cv::RANSAC);
    results.push_back(H);

    return results;
}

std::vector<cv::Mat> surf_bf(cv::Mat frame, cv::cuda::GpuMat referenceFrame, cv::cuda::GpuMat mask, cv::Ptr<cv::cuda::SURF_CUDA> detector, cv::Ptr<cv::cuda::DescriptorMatcher> matcher)
{
    cv::cuda::Stream cudaStream1;

    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

    cv::cuda::GpuMat img1_gpu, mask_gpu; // this holds the keys detected
    img1_gpu.upload(frame);

    cv::cuda::GpuMat keys1, keys2;                // this holds the keys detected
    cv::cuda::GpuMat desc1, desc2;                // this holds the descriptors for the detected keypoints
    std::vector<cv::KeyPoint> cpuKeys1, cpuKeys2; // holds keypoints downloaded from gpu
    detector->detectWithDescriptors(img1_gpu, mask_gpu, keys1, desc1, false);
    detector->downloadKeypoints(keys1, cpuKeys1); // download keys to cpu if needed for anything...like displaying or whatever
    detector->detectWithDescriptors(referenceFrame, mask_gpu, keys2, desc2, false);
    detector->downloadKeypoints(keys2, cpuKeys2); // download keys to cpu if needed for anything...like displaying or whatever

    // cv::Mat cpuDesc1(desc1), cpuDesc2(desc2);

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    // cv::Ptr<cv::DescriptorMatcher> tmp_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    // std::vector<std::vector<cv::DMatch>> knn_matches;
    // tmp_matcher->knnMatch(cpuDesc1, cpuDesc2, knn_matches, 2);
    // //-- Filter matches using the Lowe's ratio test
    // const float ratio_thresh = 0.75f;
    // std::vector<cv::DMatch> good_matches;
    // for (size_t i = 0; i < knn_matches.size(); i++)
    // {
    //     if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    //     {
    //         good_matches.push_back(knn_matches[i][0]);
    //     }
    // }


    std::vector<std::vector<cv::DMatch>> cpuKnnMatches;
    cv::cuda::GpuMat gpuKnnMatches;                                                     // holds matches on gpu
    matcher->knnMatchAsync(desc1, desc2, gpuKnnMatches, 2, cv::noArray(), cudaStream1); // find matches
    cudaStream1.waitForCompletion();

    matcher->knnMatchConvert(gpuKnnMatches, cpuKnnMatches); // download matches from gpu and put into vector<vector<DMatch>> form on cpu

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < cpuKnnMatches.size(); i++)
    {
        if (cpuKnnMatches[i][0].distance < ratio_thresh * cpuKnnMatches[i][1].distance)
        {
            good_matches.push_back(cpuKnnMatches[i][0]);
        }
    }

    //-- Draw matches
    cv::Mat img_matches;
    drawMatches(frame, cpuKeys1, cv::Mat(referenceFrame), cpuKeys2, good_matches, img_matches, cv::Scalar::all(-1),
                cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    std::vector<cv::Mat> results{img_matches};
    // std::vector<cv::Mat> results{frame};

    //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
    for (size_t i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back(cpuKeys1[good_matches[i].queryIdx].pt);
        scene.push_back(cpuKeys2[good_matches[i].trainIdx].pt);
    }
    cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC);
    results.push_back(H);

    return results;
}

int main(int argc, char const *argv[])
{
    cv::cuda::Stream cudaStream;
    // cv::VideoCapture cap("/mnt/local_data/providentia/test_recordings/videos/s40_n_far_image_raw.mp4");
    cv::VideoCapture cap("/mnt/local_data/providentia/test_recordings/videos/s50_s_far_image_raw.mp4");
    if (!cap.isOpened()) // if not success, exit program
    {
        std::cout << "Cannot open the video." << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FPS, 25);

    cv::Mat frame;
    cv::cuda::GpuMat referenceFrame, mask;
    mask.upload(cv::Mat(frame.size(), CV_8UC1, cv::Scalar(1)));
    bool isReferenceFrameSet = false;

    std::string windowName = "Dynamic Camera Stabilization";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    std::string matchingWindowName = "Matching Camera Stabilization";
    cv::namedWindow(matchingWindowName, cv::WINDOW_AUTOSIZE);

    cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(500, 1.2f, 8, 31, 0, 2, 0, 31, 20, true);
    cv::Ptr<cv::cuda::DescriptorMatcher> orb_matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    cv::Ptr<cv::cuda::SURF_CUDA> surf = cv::cuda::SURF_CUDA::create(400);
    cv::Ptr<cv::cuda::DescriptorMatcher> surf_matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
    cv::Ptr<cv::cuda::FastFeatureDetector> fast = cv::cuda::FastFeatureDetector::create();
    double calculationScaleFactor = 1;
    double renderingScaleFactor = 0.65;


    while (true)
    {
        cap >> frame;
        cv::Mat originalFrame = frame.clone();
        cv::resize(frame, frame, cv::Size(), calculationScaleFactor, calculationScaleFactor);

        if (!isReferenceFrameSet)
        {
            cv::Mat cpu_referenceFrame = frame.clone();
            cv::cvtColor(cpu_referenceFrame, cpu_referenceFrame, cv::COLOR_BGR2GRAY);
            referenceFrame.upload(cpu_referenceFrame);
            isReferenceFrameSet = true;
        }

        // std::vector<cv::Mat> flannResult = orb_bf(frame, referenceFrame, orb, orb_matcher);
        std::vector<cv::Mat> flannResult = surf_bf(frame, referenceFrame, mask, surf, surf_matcher);

        cv::Mat stabilized;
        cv::Mat H = flannResult[1];
        cv::warpPerspective(originalFrame, stabilized, H, originalFrame.size());
        int padding = 10;
        stabilized = cv::Mat(stabilized, cv::Rect(padding, padding, stabilized.cols - 2 * padding, stabilized.rows - 2 * padding));
        originalFrame = cv::Mat(originalFrame, cv::Rect(padding, padding, originalFrame.cols - 2 * padding, originalFrame.rows - 2 * padding));
        cv::hconcat(std::vector<cv::Mat>{originalFrame, stabilized}, stabilized);
        cv::resize(stabilized, stabilized, cv::Size(), renderingScaleFactor, renderingScaleFactor);

        cv::Mat flannMatching = flannResult[0];
        cv::resize(flannMatching, flannMatching, cv::Size(), renderingScaleFactor, renderingScaleFactor);

        cv::imshow(windowName, stabilized);
        cv::imshow(matchingWindowName, flannMatching);
        if ((char)cv::waitKey(1) == 27)
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
