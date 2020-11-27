#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <thread>         // std::thread
#include <deque>
#include <vector>

using namespace cv;
using namespace std;

int main()
{
    bool write = true;
    int scale_factor = 2;
    float optical_flow_alpha = 0.5f;

    std::string input_output_folder = "/mnt/nextcloud/tum/Master/5. Semester/Guided Research/videos/s40_n_cam_far/stamp/";
    std::string video_file = input_output_folder + "video.mp4";
    VideoCapture video(video_file);
    if (!video.isOpened())
    {
        //error in opening the video input
        cerr << "Unable to open " << video_file << endl;
        return 0;
    }
    std::string optical_flow_file = input_output_folder + "/optical_flow/dense_optical_flow.mp4";
    VideoCapture optical_flow(optical_flow_file);
    if (!optical_flow.isOpened())
    {
        //error in opening the video input
        cerr << "Unable to open " << optical_flow_file << endl;
        return 0;
    }

    Mat video_frame, optical_flow_frame;
    // skip first optical flow frame
    // optical_flow >> optical_flow_frame;

    while (true)
    {
        video >> video_frame;
        optical_flow >> optical_flow_frame;
        if (video_frame.empty() || optical_flow_frame.empty()){
            break;
        }
        resize(video_frame, video_frame, Size(1920, 1200) / scale_factor);
        resize(optical_flow_frame, optical_flow_frame, Size(1920, 1200) / scale_factor);

        // video_frame.push_back(optical_flow_frame);
        addWeighted(video_frame, 1 - optical_flow_alpha, optical_flow_frame, optical_flow_alpha, 0.0, video_frame);
        
        imshow("Comparison", video_frame);
        waitKey(3);
    }
    
    return 0;
}