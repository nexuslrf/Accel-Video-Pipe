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
#include <tuple>
#include <queue>
#include <cmath>
#include <chrono>
#include <mutex>
#include <condition_variable>

namespace avp {

using SizeVector = std::vector<size_t>;
using Tensor = torch::Tensor;
using Mat = cv::Mat;
using string = std::string;

enum PackType {
    AVP_MAT = 0,
    AVP_TENSOR = 1,
};

size_t streamCapacity = 5;
size_t numThreads = 1;

class StreamPacket{
public:
    std::vector<Mat> matList;
    std::vector<Tensor> tensorList;
    Mat mat_data;
    Tensor tensor_data;
    int timestamp, numConsume;
    PackType dataType;
    bool finish{false};

    StreamPacket(PackType data_type=AVP_TENSOR, int init_timestamp=-1): timestamp(init_timestamp), 
        numConsume(0), dataType(data_type) {}
    StreamPacket(Tensor& tensor_data, int tensor_timestamp=-1): 
        timestamp(tensor_timestamp), numConsume(0), dataType(AVP_TENSOR) 
    {
        tensorList.push_back(tensor_data);
    }
    StreamPacket(Mat& mat_data, int mat_timestamp=-1):
        timestamp(mat_timestamp), numConsume(0), dataType(AVP_MAT) 
    {
        matList.push_back(mat_data);
    }
    bool empty()
    {
        if(dataType==AVP_TENSOR)
            return tensorList.empty() || tensorList.front().numel() == 0;
        else if(dataType==AVP_MAT)
            return matList.empty() || matList.front().empty();
        else
            return true;
    }

    bool empty(PackType data_type)
    {
        if(data_type==AVP_TENSOR)
            return tensorList.empty() || tensorList.front().numel() == 0;
        else if(data_type==AVP_MAT)
            return matList.empty() || matList.front().empty();
        else
            return true;
    }

    void* data_ptr(int idx=0)
    {
        if(dataType==AVP_MAT)
            return matList[idx].data;
        else if(dataType==AVP_TENSOR)
            return tensorList[idx].data_ptr();
        return NULL;
    }

    Mat& mat(size_t idx=0, bool enlist=false)
    {
        if(dataType==AVP_MAT)
        {
            if(idx < matList.size())
            {
                return matList[idx];
            }
            else
            {
                matList.push_back(mat_data);
                return matList.front();
            }
        }

        else
        {   // tensor to mat
            if(!empty(AVP_MAT)||empty(AVP_TENSOR))
            {   
                if(enlist)
                    matList.push_back(mat_data);
                return mat_data;
            }
            else
            {
                // @TODO: must ensure mat.type()==CV_32FC3
                // tensor's data layout must be HWC rather than CHW.
                // ignore type checking here...
                // TensorToMat is not that common compared to MatToTensor...
                tensor_data = tensorList[idx];
                mat_data = cv::Mat(tensor_data.size(0), tensor_data.size(1), CV_32FC3, tensor_data.data_ptr());
                if(enlist)
                    matList.push_back(mat_data);
                return mat_data;
            }
            
        }
    }
    Tensor& tensor(size_t idx=0, bool enlist=false)
    {
        if(dataType==AVP_TENSOR)
        {
            if(idx < tensorList.size())
            {
               return tensorList[idx]; 
            }
            else
            {
                tensorList.push_back(tensor_data);
                return tensorList.front();
            }
        }
        else
        {   // mat to tensor, no memcpy
            if(empty(AVP_MAT)||!empty(AVP_TENSOR))
            {
                if(enlist)
                    tensorList.push_back(tensor_data);
                return tensor_data;
            }
            else
            {
                // @TODO: must ensure mat.type()==CV_32FC3
                // ignore type checking here...
                mat_data = matList[idx];
                tensor_data = torch::from_blob(mat_data.data, {mat_data.rows, mat_data.cols, 3}, torch::kFloat32);
                if(enlist)
                    tensorList.push_back(tensor_data);
                return tensor_data;
            }
            
        }
    }
    void loadData(const Tensor& t_data)
    {
        tensorList.push_back(t_data);
    }
    void loadData(const Mat& m_data)
    {
        matList.push_back(m_data);
    }
    size_t size()
    {
        if(dataType==AVP_MAT)
            return matList.size();
        else
            return tensorList.size();
    }
    size_t size(PackType data_type)
    {
        if(data_type==AVP_MAT)
            return matList.size();
        else
            return tensorList.size();
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
    std::condition_variable loadCond, spaceCond;
public:
    std::string name;
    int numConsume;
    std::vector<Stream*> coupledStreams;

    Stream(): numConsume(0) {}
    Stream(std::string s_name, int num_consume=0): name(s_name), numConsume(num_consume)
    {}
    void loadPacket(StreamPacket& packet, int num_consume=0) 
    {
        if(num_consume<=0)
            num_consume = numConsume;
        for(auto &ptr: coupledStreams)
            ptr->loadPacket(packet);
        std::unique_lock<std::mutex> locker(consumeMutex);
        spaceCond.wait(locker, [&](){return this->size()<=streamCapacity;});
        push_back(packet);
        back().numConsume = num_consume;
        locker.unlock();
        loadCond.notify_one();
    }
    //@TODO: Consume + while(!queue.empty())...
    void releasePacket(iterator& it)
    {
        {
            std::lock_guard<std::mutex> guard(consumeMutex);
            it->numConsume--;
            if(!it->numConsume&&it==begin())
            {
                pop_front();
            }
        }
        spaceCond.notify_one();
    }
    void releasePacket()
    {
        {
            std::lock_guard<std::mutex> guard(consumeMutex);
            auto it = begin();
            it->numConsume--;
            if(!it->numConsume)
                pop_front();
        }
        spaceCond.notify_one();
    }
    void coupleStream(std::vector<Stream*> stream_ptr_list)
    {
        for(auto &ptr: stream_ptr_list)
        {
            coupledStreams.push_back(ptr);
        }
    }
    StreamPacket& getPacket()
    {
        std::unique_lock<std::mutex> locker(consumeMutex);
        loadCond.wait(locker, [this](){ return !this->empty();} );
        // consider release packet here?
        locker.unlock();
        return front();
    }
};

const int MAX_TIME_ROUND = 100000000;
using DataList = std::vector<StreamPacket>;

class PipeProcessor{
public:
    std::string name;
    // @TODO: vector or map ?
    std::vector<Stream*> inStreams, outStreams;
    PPType procType;
    PackType dataType; // only used for output data type
    size_t numInStreams, numOutStreams;
    int timeTick;
    bool skipEmptyCheck, finish; // used to skip empty checking, to enable proper functioning

#ifdef _TIMING
    int cumNum{0};
    double averageMeter{0};
#endif

    PipeProcessor(int num_instreams, int num_outstreams, PackType data_type, std::string pp_name, PPType pp_type): name(pp_name), 
        procType(pp_type), dataType(data_type), numInStreams(num_instreams), numOutStreams(num_outstreams), 
        timeTick(-1), skipEmptyCheck(false), finish(false)
    {}
    void addTick() {
        timeTick = (timeTick + 1) % MAX_TIME_ROUND;
    }
    virtual void run(DataList& in_data_list, DataList& out_data_list) {}
    virtual void process() 
    {
        checkStream();

        DataList in_data_list, out_data_list;
        int tmp_time=-2;
        bool packetEmpty = false; // streamEmpty = false, timeInconsistent = false;
        bool pipeFinish = false;
        size_t i;
        for(i=0; i<numInStreams; i++)
        {
            // Right now just assume all streams are coherent.
            StreamPacket in_data;

            in_data = inStreams[i]->getPacket();
            pipeFinish = pipeFinish | in_data.finish;
            if(tmp_time==-2)
                tmp_time = in_data.timestamp;
            else if(tmp_time != in_data.timestamp)
            {
                //timeInconsistent = true; // Case checking 2
#ifdef _LOG_INFO                    
                std::cerr<<"[WARNING] "<<typeid(*this).name()<<" inconsistent timestamps of inStream packets\n";
#endif                    
                exit(1);//return;
            }

            if(!skipEmptyCheck && in_data.empty())
            {
                packetEmpty = true; // Case checking 3
            }
            else
                in_data_list.push_back(in_data); 
                // No matter what case, in_data_list will be generated
        }
        
        finish = pipeFinish;

        if(timeTick == tmp_time)
        {
            // @TODO other ways!
#ifdef _LOG_INFO
            std::cerr<<"[WARNING] This packet has been computed!\n";
#endif
            return;
        }

        if(numInStreams)
            timeTick = tmp_time;
        assert(timeTick >= 0);
        //  timeTick >= 0;

        for(i=0; i<numOutStreams; i++)
        {
            StreamPacket out_data(dataType, timeTick);
            if(pipeFinish)
            {
                out_data.finish = true;
                outStreams[i]->loadPacket(out_data);
            }
            else if(packetEmpty)
            {
                outStreams[i]->loadPacket(out_data);
            }
            else
            {
                out_data_list.push_back(out_data);
            }
            
        }
        
        if(packetEmpty || pipeFinish) // clean up all inStream packets
        {
#ifdef _LOG_INFO            
            std::cerr<<"[WARNING] "<<typeid(*this).name()<<" clean up all inStream packets\n";
#endif            
            for(i=0; i<numInStreams; i++)
            {
                inStreams[i]->releasePacket();
            }
            return;
        }

#ifdef _TIMING
        auto stop1 = std::chrono::high_resolution_clock::now();
#endif

        run(in_data_list, out_data_list);

#ifdef _TIMING
        auto stop2 = std::chrono::high_resolution_clock::now();
        auto gap = std::chrono::duration_cast<std::chrono::milliseconds>(stop2-stop1).count();
        averageMeter = ((averageMeter * cumNum) + gap) / (cumNum + 1);
        cumNum++;
#endif

        for(i=0; i<numOutStreams; i++)
            outStreams[i]->loadPacket(out_data_list[i]);
        for(i=0; i<numInStreams; i++)
            inStreams[i]->releasePacket();
    }
    
    virtual void bindStream(std::vector<Stream*> in_stream_ptr_list, std::vector<Stream*> out_stream_ptr_list) 
    {
        for(auto& stream_ptr: in_stream_ptr_list)
        {
            if(inStreams.size()==numInStreams)
            {
#ifdef _LOG_INFO
                std::cerr<<"[ERROR] "<<typeid(*this).name()<<" Number of inStreams exceeds limit.\n";
#endif
                exit(0);
            }
            inStreams.push_back(stream_ptr);
            stream_ptr->numConsume++;
        }
        for(auto& stream_ptr: out_stream_ptr_list)
        {
            if(outStreams.size()==numOutStreams)
            {
#ifdef _LOG_INFO                
                std::cerr<<"[ERROR] "<<typeid(*this).name()<<" Number of outStreams exceeds limit.\n";
#endif                
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
#ifdef _LOG_INFO            
            std::cerr<<"[ERROR] "<<typeid(*this).name()<<" Streams are empty!\n";
#endif            
            exit(0);
        }
    }
    bool checkEmpty(int start=0, int end=-1)
    {
        if(end==-1)
            end = inStreams.size();
        for(int i=start; i<end; i++)
        {
            if(inStreams[i]->empty()||inStreams[i]->front().empty())
                return true;
        }
        return false;
    }
    // virtual void Stop() = 0;
};

}

    // StreamPacket& getPacket()
    // {
    //     bool is_empty = empty();
    //     std::__1::chrono::steady_clock::time_point stop1;
    //     if(is_empty)
    //     {
    //         std::cout<<"Wait for Packet!\n";
    //         stop1 = std::chrono::high_resolution_clock::now();
    //     }
    //     std::unique_lock<std::mutex> locker(consumeMutex);
    //     loadCond.wait(locker, [this](){ return !this->empty();} );
    //     // consider release packet here?
    //     locker.unlock();
    //     if(is_empty)
    //     {
    //         auto stop2 = std::chrono::high_resolution_clock::now();
    //         auto gap = std::chrono::duration_cast<std::chrono::milliseconds>(stop2-stop1).count();
    //         std::cout<<"Time spent:" <<gap<<" ms\n";
    //     }
    //     return front();
    // }