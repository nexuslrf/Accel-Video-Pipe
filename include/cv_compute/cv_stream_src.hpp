#pragma once

#include <opencv2/opencv.hpp>
#include "../avpipe/base.hpp"

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
    bool flip;
    int rawWidth, rawHeight, fps;
    StreamSrcProcessor(std::string pp_name, bool img_flip, SRC_MODE src_type): PipeProcessor(0, 1, AVP_MAT, pp_name, STREAM_INIT)
    {
        srcType = src_type;
        timeTick = 0;
        flip = img_flip;
    }

    void run(DataList& in_data_list, DataList& out_data_list)
    {
        Mat frame;
        cap>>frame;
        if(!frame.empty())
        {
            if(flip)
                cv::flip(frame, frame, +1);
            out_data_list[0].loadData(frame);
        }
        else
        {
            out_data_list[0].finish = true;
            finish = true;
        }
        addTick();
    }

};

class VideoFileProcessor: public StreamSrcProcessor {
public:
    VideoFileProcessor(std::string file_path, bool img_flip=false, std::string pp_name="VideoFileProcessor"): 
        StreamSrcProcessor(pp_name, img_flip, VIDEO_FILE)
    {
        cap.open(file_path);
        fps = cap.get(cv::CAP_PROP_FPS);
        rawWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        rawHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    }
};

class WebCamProcessor: public StreamSrcProcessor {
public:
    WebCamProcessor(int cam_id= 0, bool img_flip=false, std::string pp_name="WebCamProcessor"): 
        StreamSrcProcessor(pp_name, img_flip, WEB_CAM)
    {
        cap.open(cam_id);
        fps = cap.get(cv::CAP_PROP_FPS);
        rawWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        rawHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    }
};

class ImgFileProcessor: public PipeProcessor {
    std::string filePath;
    bool flip, inputMode;
public:
    int rawWidth, rawHeight;
    ImgFileProcessor(std::string file_path, bool img_flip=false, bool input_mode=false, std::string pp_name="ImgFileProcessor"):
        filePath(file_path), inputMode(input_mode), PipeProcessor(0, 1, AVP_MAT, pp_name, STREAM_INIT)
    {
        auto frame = cv::imread(file_path);
        rawHeight = frame.rows;
        rawWidth = frame.cols;
        flip = img_flip;
        timeTick = 0;
    }
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        Mat frame;
        string inputFile;
        if(inputMode)
            std::cin>>inputFile;
        if(inputFile!="")
            frame = cv::imread(inputFile);
        else
            frame = cv::imread(filePath);
        if(!frame.empty())
        {
            // if(BGR2RGB)
            //     cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
            if(flip)
                cv::flip(frame, frame, +1);
            out_data_list[0].loadData(frame);
        }
        else
        {
            out_data_list[0].finish = true;
            finish = true;
        }
        addTick();
    }
};

}