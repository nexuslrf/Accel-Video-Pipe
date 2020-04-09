#pragma once

#include "../avpipe/base.hpp"
namespace avp {

class StreamShowProcessor: public PipeProcessor {
public:
    int waitTime;
    string showName;
    StreamShowProcessor(int wait_time=1, string show_name="", string pp_name=""): 
        PipeProcessor(pp_name, STREAM_SINK), waitTime(wait_time), showName(show_name) {}
    void Process()
    {
        checkStream();
        auto in_data = inStreams[0]->front();
        if(in_data.empty())
        {
            inStreams[0]->ReleasePacket();
            return;
        } 
        if(in_data.timestamp==timeTick)
            return;
        else
            timeTick = in_data.timestamp;
        cv::imshow(showName, in_data.mat);
        cv::waitKey(waitTime);
        inStreams[0]->ReleasePacket();
    }
};

}