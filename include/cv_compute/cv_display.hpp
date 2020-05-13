#pragma once

#include "../avpipe/base.hpp"
namespace avp {

class StreamShowProcessor: public PipeProcessor {
public:
    int waitTime;
    string showName;
    StreamShowProcessor(int wait_time=1, string show_name="", string pp_name="streamShowProcessor"): 
        PipeProcessor(1, 0, AVP_MAT, pp_name, STREAM_SINK), waitTime(wait_time), showName(show_name) {}
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        // std::cout<<"Run Show: "<<in_data_list.front().mat.empty()<<std::endl;
        cv::imshow(showName, in_data_list[0].mat());
        cv::waitKey(waitTime);
    }
};

}