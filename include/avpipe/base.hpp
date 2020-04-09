/*!
 * Copyright (c) 2020 by Contributors
 * author Ruofan Liang
 * base data structure for AV-Pipe
 * Suppose default stream data type is torch::tensor
 */
#pragma once

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <mutex>

namespace avp {

using SizeVector = std::vector<size_t>;
using Tensor = torch::Tensor;
using Mat = cv::Mat;

enum PackType {
    AVP_MAT = 0,
    AVP_TENSOR = 1
};
class StreamPacket{
public:
    Mat mat;
    Tensor tensor;
    int timestamp;
    int numConsume;
    PackType dataType;
    StreamPacket(PackType data_type=AVP_TENSOR, int init_timestamp=-1): timestamp(init_timestamp), numConsume(0), dataType(data_type) {}
    StreamPacket(Tensor& tensor_data, int tensor_timestamp=-1): 
        tensor(tensor_data), timestamp(tensor_timestamp), numConsume(0), dataType(AVP_TENSOR) {}
    StreamPacket(Mat& mat_data, int mat_timestamp=-1):
        mat(mat_data), timestamp(mat_timestamp), numConsume(0), dataType(AVP_MAT) {}
    bool empty(PackType data_type=AVP_TENSOR)
    {
        if(data_type==AVP_TENSOR)
            return tensor.numel() == 0;
        else if(data_type==AVP_MAT)
            return mat.empty();
        else
            return true;
    }
    void* data_ptr()
    {
        if(dataType==AVP_MAT)
            return mat.data;
        else if(dataType==AVP_TENSOR)
            return tensor.data_ptr();
        return NULL;
    }

    // Mat getMat(){}
    // Tensor getTensor(){}
};

/*! pipe processor type */
enum PPType {
    STREAM_PROC = 0,
    STREAM_INIT = 1,
    STREAM_SINK = 2
};
enum StreamType {
    AVP_STREAM_IN = 0,
    AVP_STREAM_OUT = 1
};

class Stream: public std::deque<StreamPacket>{
    std::mutex consumeMutex;
public:
    std::string name;
    int numConsume;
    Stream(): numConsume(0) {}
    Stream(std::string s_name, int num_consume=0): name(s_name), numConsume(num_consume)
    {}
    void LoadPacket(StreamPacket& packet) 
    {
        packet.numConsume = numConsume;
        push_back(packet);
    }
    //@TODO: Consume + while(!queue.empty())...
    void ReleasePacket(iterator& it)
    {
        std::lock_guard<std::mutex> guard(consumeMutex);
        it->numConsume--;
        if(!it->numConsume&&it==begin())
        {
            pop_front();
        }
    }
    void ReleasePacket()
    {
        auto it = begin();
        std::lock_guard<std::mutex> guard(consumeMutex);
        it->numConsume--;
        if(!it->numConsume)
            pop_front();
    }
};

#define MAX_TIME_ROUND 100000000

class PipeProcessor{
public:
    std::string name;
    // @TODO: vector or map ?
    std::vector<Stream*> inStreams, outStreams;
    std::vector<Stream::iterator> inIterators, outInterators;
    PPType procType;
    int timeTick;
    PipeProcessor(std::string pp_name, PPType pp_type): name(pp_name), procType(pp_type), timeTick(-1)
    {}
    void AddTick() {
        timeTick = (timeTick + 1) % MAX_TIME_ROUND;
    }
    virtual void Process() {}
    virtual void BindStream(Stream* stream_ptr, StreamType stream_type) 
    {
        if(stream_type==AVP_STREAM_IN)
        {
            inStreams.push_back(stream_ptr);
            stream_ptr->numConsume++;
        }
        else if(stream_type==AVP_STREAM_OUT)
            outStreams.push_back(stream_ptr);
    }
    // virtual void Stop() = 0;
};
}
