#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <thread>         // std::thread
#include <deque>
#include <vector>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>

using namespace cv;
using namespace std;

Mat render_optical_flow(Mat prvs_color, Mat next_color, float optical_flow_alpha)
{
    Mat prvs, next;
    cvtColor(next_color, next, COLOR_BGR2GRAY);
    cvtColor(prvs_color, prvs, COLOR_BGR2GRAY);

    Mat flow(prvs.size(), CV_32FC2);
    calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    // visualization
    Mat flow_parts[2];
    split(flow, flow_parts);
    Mat magnitude, angle, magn_norm;
    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));
    //build hsv image
    Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cvtColor(hsv8, bgr, COLOR_HSV2BGR);

    resize(bgr, bgr, prvs_color.size());
    addWeighted(prvs_color, 1 - optical_flow_alpha, bgr, optical_flow_alpha, 0.0, bgr);
    return bgr;
}

std::stringstream create_directories(std::string input_output_folder, float optical_flow_alpha){
    std::string alpha_string = std::to_string(optical_flow_alpha);
    int index = alpha_string.rfind('.');
    alpha_string.erase(alpha_string.begin()+index);
    std::stringstream filename;
    filename << input_output_folder << "optical_flow/";
    mkdir(filename.str().c_str(), 0777);
    filename << "alpha_" << alpha_string;
    mkdir(filename.str().c_str(), 0777);
    return filename;
}

void write_optical_flow(Mat prvs, Mat next, float optical_flow_alpha, std::string input_output_folder, int i)
{
    Mat bgr = render_optical_flow(prvs, next, optical_flow_alpha);
    
    std::stringstream filename = create_directories(input_output_folder, optical_flow_alpha);
    filename << "/image_" << i << ".png"; 

    imwrite(filename.str(), bgr);
}

void test_thread_func(int i){
    std::cout << "Yeet: " << std::to_string(i) << std::endl;
}

int main()
{
    std::deque<std::thread> write_thread_pool;
    float optical_flow_alpha = 0.75f;

    bool write = true;
    std::string input_output_folder = "/mnt/nextcloud/tum/Master/5. Semester/Guided Research/videos/s40_n_cam_far/stamp/";
    VideoCapture capture(input_output_folder + "video.mp4");
    if (!capture.isOpened())
    {
        //error in opening the video input
        cerr << "Unable to open file!" << endl;
        return 0;
    }
    int scale_factor = 1;
    Mat prvs, next;
    capture >> prvs;

    int i = 0;
    while (true)
    {
        capture >> next;
        if (next.empty() || prvs.empty())
            break;

        resize(prvs, prvs, next.size() / scale_factor);
        resize(next, next, next.size() / scale_factor);

        if (write)
        {
            // write_optical_flow(prvs, next, input_output_folder, i++);
            // write_thread_pool.push_back(std::thread(test_thread_func, i++));
            // write_optical_flow(prvs.clone(), next.clone(), optical_flow_alpha, input_output_folder, i++);
            write_thread_pool.push_back(std::thread(write_optical_flow, prvs.clone(), next.clone(), optical_flow_alpha, input_output_folder, i++));
            if(write_thread_pool.size() >= 12){
                write_thread_pool.front().join();
                write_thread_pool.pop_front();
            }
            // std::thread write_thread(write_optical_flow, prvs, next, input_output_folder, i++);
        }
        else
        {
            Mat bgr = render_optical_flow(prvs.clone(), next.clone(), optical_flow_alpha);
            imshow("dense optical flow", bgr);
       
            int keyboard = waitKey(30);
            if (keyboard == 'q' || keyboard == 27)
                break;
        }
        prvs = next.clone();
    }
}