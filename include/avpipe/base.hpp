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
using string = std::string;

enum PackType {
    AVP_MAT = 0,
    AVP_TENSOR = 1
};
class StreamPacket{
public:
    Mat mat;
    Tensor tensor;
    int timestamp, numConsume, batchSize;
    PackType dataType;
    StreamPacket(PackType data_type=AVP_TENSOR, int init_timestamp=-1): timestamp(init_timestamp), numConsume(0), 
        batchSize(1), dataType(data_type) {}
    StreamPacket(Tensor& tensor_data, int tensor_timestamp=-1): 
        tensor(tensor_data), timestamp(tensor_timestamp), numConsume(0), batchSize(1), dataType(AVP_TENSOR) {}
    StreamPacket(Mat& mat_data, int mat_timestamp=-1):
        mat(mat_data), timestamp(mat_timestamp), numConsume(0), batchSize(1), dataType(AVP_MAT) {}
    bool empty()
    {
        if(dataType==AVP_TENSOR)
            return tensor.numel() == 0;
        else if(dataType==AVP_MAT)
            return mat.empty();
        else
            return true;
    }
    bool empty(PackType data_type)
    {
        if(data_type==AVP_TENSOR)
            return tensor.numel() == 0;
        else if(data_type==AVP_MAT)
            return mat.empty();
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

    Mat& matData()
    {
        if(dataType==AVP_MAT)
            return mat;
        else
        {   // tensor to mat
            if(!mat.empty()||tensor.numel()==0)
                return mat;
            else
            {
                // @TODO: must ensure mat.type()==CV_32FC3
                // tensor's data layout must be HWC rather than CHW.
                // ignore type checking here...
                // TensorToMat is not that common compared to MatToTensor...
                mat = cv::Mat(tensor.size(0), tensor.size(1), CV_32FC3, tensor.data_ptr());
                return mat;
            }
            
        }
    }
    Tensor& tensorData()
    {
        if(dataType==AVP_TENSOR)
            return tensor;
        else
        {   // mat to tensor, no memcpy
            if(mat.empty()||tensor.numel()!=0)
                return tensor;
            else
            {
                // @TODO: must ensure mat.type()==CV_32FC3
                // ignore type checking here...
                tensor = torch::from_blob(mat.data, {mat.rows, mat.cols, 3}, torch::kFloat32);
                return tensor;
            }
            
        }
    }
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
    void loadPacket(StreamPacket& packet) 
    {
        packet.numConsume = numConsume;
        push_back(packet);
    }
    //@TODO: Consume + while(!queue.empty())...
    void releasePacket(iterator& it)
    {
        std::lock_guard<std::mutex> guard(consumeMutex);
        it->numConsume--;
        if(!it->numConsume&&it==begin())
        {
            pop_front();
        }
    }
    void releasePacket()
    {
        auto it = begin();
        std::lock_guard<std::mutex> guard(consumeMutex);
        it->numConsume--;
        if(!it->numConsume)
            pop_front();
    }
};

const int MAX_TIME_ROUND = 100000000;
using DataList = std::vector<StreamPacket>;

class PipeProcessor{
public:
    std::string name;
    // @TODO: vector or map ?
    std::vector<Stream*> inStreams, outStreams;
    std::vector<Stream::iterator> iterators;
    PPType procType;
    PackType dataType; // only used for output data type
    size_t numInStreams, numOutStreams;
    int timeTick;
    PipeProcessor(int num_instreams, int num_outstreams, PackType data_type, std::string pp_name, PPType pp_type): name(pp_name), 
        procType(pp_type), dataType(data_type), numInStreams(num_instreams), numOutStreams(num_outstreams), timeTick(-1)
    {}
    void addTick() {
        timeTick = (timeTick + 1) % MAX_TIME_ROUND;
    }
    virtual void run(DataList& in_data_list, DataList& out_data_list) {}
    virtual void process() 
    {
        checkStream();
        DataList in_data_list, out_data_list;
        bool finish = false;
        size_t i;
        for(i=0; i<numInStreams; i++)
        {
            // @TODO: you may need to think about time sync here! 
            // Right now just assume all streams are coherent.
            if(inStreams[i]->empty())
                return;
            auto in_data = inStreams[i]->front();
            if(i==0)
            {
                int tmp_time = in_data.timestamp;
                if(tmp_time==timeTick)
                    return;
                else
                    timeTick = tmp_time;
            }

            if(in_data.empty()||finish)
            {
                finish = true;
                inStreams[i]->releasePacket();
            }
            else
                in_data_list.push_back(in_data);
        }
        for(i=0; i<numOutStreams; i++)
        {
            StreamPacket out_data(dataType, timeTick);
            if(finish)
            {
                outStreams[i]->loadPacket(out_data);
            }
            else
            {
                out_data_list.push_back(out_data);
            }
            
        }
        
        if(finish)
            return;
        
        run(in_data_list, out_data_list);

        for(i=0; i<numOutStreams; i++)
            outStreams[i]->loadPacket(out_data_list[i]);
        for(i=0; i<numInStreams; i++)
            inStreams[i]->releasePacket();
    }
    virtual void bindStream(Stream* stream_ptr, StreamType stream_type) 
    {
        if(stream_type==AVP_STREAM_IN)
        {
            if(inStreams.size()==numInStreams)
            {
                std::cerr<<"[ERROR] "<<typeid(*this).name()<<" Number of inStreams exceeds limit.\n";
                exit(0);
            }
            inStreams.push_back(stream_ptr);
            stream_ptr->numConsume++;
        }
        else if(stream_type==AVP_STREAM_OUT)
        {
            if(outStreams.size()==numOutStreams)
            {
                std::cerr<<"[ERROR] "<<typeid(*this).name()<<" Number of outStreams exceeds limit.\n";
                exit(0);
            }
            outStreams.push_back(stream_ptr);
        }
    }
    void checkStream()
    {
        // @TODO: Not sufficient!
        if(inStreams.empty()&&outStreams.empty())
        {
            std::cerr<<"[ERROR] "<<typeid(*this).name()<<" Streams are empty!\n";
            exit(0);
        }
    }
    // virtual void Stop() = 0;
};
}
