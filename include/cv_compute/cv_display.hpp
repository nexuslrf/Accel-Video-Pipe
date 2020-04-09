#pragma once

#include "../avpipe/base.hpp"
namespace avp {

class StreamShowProcessor: public PipeProcessor {
public:
    string showName;
    StreamShowProcessor(string show_name, string pp_name): 
        PipeProcessor(pp_name, STREAM_SINK), showName(show_name) {}
    void Process()
    {
        checkStream();
        auto in_data = inStreams[0]->front();
        if(in_data.empty() || in_data.timestamp==timeTick)
            return;
        else
            timeTick = in_data.timestamp;
        cv::imshow("PalmDetection", in_data.mat);
    }
};

}