
#include "glog/logging.h"
#include "ceres/ceres.h"
#include "lib/optimization_problems.hpp"
#include "curve_fitting_data.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <iostream>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::NumericDiffCostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
const char* keys =
    "{ help h |                  | Print help message. }"
    "{ input1 | box.png          | Path to input image 1. }"
    "{ input2 | box_in_scene.png | Path to input image 2. }";

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    
    minimizeHelloWorldProblem(0.5);
    minimizePowellsFunction(3.0, -1.0, 0.0, 1.0);
    minimizeExponentialFunction(0.0, 0.0, kNumObservations, data);

    return 0;
}