#include <stdio.h>
#include <iostream>
#include "lib/CameraStabilization/CameraStabilization.hpp"
#include "lib/ImageUtils/ImageUtils.hpp"
#include <opencv2/opencv.hpp>

int main(int argc, char const *argv[])
{
    std::string bridge = "s50_s";
    std::string near_or_far = "far";
    std::stringstream base_path;
    base_path << "/mnt/local_data/providentia/test_recordings/images/" << bridge << "_cam_" << near_or_far << "/stamp/";
    base_path << "1598434161.772116137.png";
    std::cout << base_path.str() << std::endl;

    cv::Mat image;
    image = cv::imread(base_path.str(), 1 );
    ImageBasedVideoCapture cap("test");

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image);
    cv::waitKey(0);


    return 0;
}
