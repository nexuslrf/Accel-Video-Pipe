#pragma once

#include <opencv2/opencv.hpp>
#include "../avpipe/base.hpp"
#include "../avpipe/cvmat_package.hpp"

namespace avp {

enum SRC_MODE {
    VIDEO_FILE = 0,
    WEB_CAM = 1
};

class StreamSrcProcessor: public PipeProcessor {
    SRC_MODE srcType;
    Mat frame;
public:
    cv::VideoCapture cap;
    int rawWidth, rawHeight, fps;
    StreamSrcProcessor(std::string pp_name, SRC_MODE src_type): PipeProcessor(pp_name, STREAM_INIT)
    {
        srcType = src_type;
        timeTick = 0;
    }
    void Process()
    {
        checkStream();
        StreamPacket out_data(AVP_MAT, timeTick);
        AddTick();
        cap>>frame;
        out_data.mat = frame;
        outStreams[0]->LoadPacket(out_data);
    }
};

class VideoFileProcessor: public StreamSrcProcessor {
public:
    VideoFileProcessor(std::string file_path, std::string pp_name=""): 
        StreamSrcProcessor(pp_name, VIDEO_FILE)
    {
        cap.open(file_path);
        fps = cap.get(cv::CAP_PROP_FPS);
        rawWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        rawHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    }
};

class WebCamProcessor: public StreamSrcProcessor {
public:
    WebCamProcessor(int cam_id= 0, std::string pp_name=""): 
        StreamSrcProcessor(pp_name, WEB_CAM)
    {
        cap.open(cam_id);
        fps = cap.get(cv::CAP_PROP_FPS);
        rawWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        rawHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    }
};

}