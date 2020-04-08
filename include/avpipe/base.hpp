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

namespace avp {

using SizeVector = std::vector<size_t>;
using Tensor = torch::Tensor;
using Mat = cv::Mat;

enum PackType {
    AVP_MAT = 0,
    AVP_TENSOR = 1
};
class StreamPackage{
public:
    Mat mat;
    Tensor tensor;
    int timestamp;
    bool empty;
    PackType dataType;
    StreamPackage(PackType data_type=AVP_TENSOR): timestamp(-1), empty(1), dataType(data_type) {}
    StreamPackage(Tensor& tensor_data, int tensor_timestamp=-1): 
        tensor(tensor_data), timestamp(tensor_timestamp), dataType(AVP_TENSOR)
    {
        empty = tensor.numel() == 0?true:false;
    }
    StreamPackage(Mat& mat_data, int mat_timestamp=-1):
        mat(mat_data), timestamp(mat_timestamp), dataType(AVP_MAT)
    {
        empty = mat.empty()?true:false;
    }
    void* data_ptr()
    {
        if(dataType==AVP_MAT)
            return mat.data;
        else if(dataType==AVP_TENSOR)
            return tensor.data_ptr();
    }
    Mat getMat(){}
    Tensor getTensor(){}
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

class Stream: public std::queue<StreamPackage>{
public:
    std::string name;
    int numConsume;
    Stream(): numConsume(0) {}
    Stream(std::string s_name, int num_consume=0): name(s_name), numConsume(num_consume)
    {}
    //@TODO: Consume + while(!queue.empty())...
    void Consume()
    {
        if(numConsume)
        {
            numConsume--;
        }
        if(!numConsume)
        {
            pop();
        }
    }
};

class PipeProcessor{
public:
    std::string name;
    // @TODO: vector or map ?
    std::vector<Stream*> inStreams, outStreams;
    PPType procType;
    PipeProcessor(std::string pp_name, PPType pp_type): name(pp_name), procType(pp_type)
    {}
    virtual void Process() {}
    virtual void BindStream(Stream* stream_ptr, StreamType stream_type) 
    {
        if(stream_type==AVP_STREAM_IN)
            inStreams.push_back(stream_ptr);
        else if(stream_type==AVP_STREAM_OUT)
            outStreams.push_back(stream_ptr);
    }
    // virtual void Stop() = 0;
};
}
