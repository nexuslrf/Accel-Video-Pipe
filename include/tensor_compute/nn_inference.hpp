/*!
 * Copyright (c) 2020 by Contributors
 * author Ruofan Liang
 * neural network interface for AV-Pipe
 * Suppose all inData are NCHW format. 
 * TODO: add NCHW <-> NHWC transformation func.
 */
#pragma once

#include "../avpipe/base.hpp"
namespace avp {

enum IEType {
    ONNX_RT = 0,
    OPENVINO = 1,
    LIBTORCH = 2
};

enum DataLayout {
    NHWC = 0,
    NCHW = 1
};

class NNProcessor: public PipeProcessor{
public:
    size_t inWidth;
    size_t inHeight;
    size_t batchSize;
    size_t channels;
    IEType ieType;
    DataLayout dataLayout;
    NNProcessor(SizeVector dims, IEType ie_type, DataLayout data_layout, int num_output,
                std::string pp_name): PipeProcessor(1, num_output, AVP_TENSOR, pp_name, STREAM_PROC)
    {
        batchSize = dims[0]; // -1 or 0 means resizable
        channels = dims[1];
        inHeight = dims[2];
        inWidth = dims[3];
        ieType = ie_type;
        dataLayout = data_layout;
    }


    /* Below are deprecated!!! */
    virtual void infer(StreamPacket& in_data, StreamPacket& out_data) {};

    void process_Deprecated()
    {
        checkStream();
        if(inStreams[0]->empty())
            return;
        auto in_data = inStreams[0]->front();
        StreamPacket out_data(AVP_TENSOR, in_data.timestamp);
        if(in_data.empty())
        {
            inStreams[0]->releasePacket();
            outStreams[0]->loadPacket(out_data);
            return;
        }
        if(in_data.timestamp==timeTick)
            return;
        else
            timeTick = in_data.timestamp;
        infer(in_data, out_data);
        outStreams[0]->loadPacket(out_data);
        inStreams[0]->releasePacket();
    }
    void process_()
    {
        checkStream();
        // @TODO: disentangle while loop in the future!
        while(!inStreams[0]->empty())
        {
            auto in_data = inStreams[0]->front();
            if(in_data.empty())
            {
                inStreams[0]->releasePacket();
                break;
            }
            if(in_data.timestamp==timeTick)
                continue;
            else
                timeTick = in_data.timestamp;
            StreamPacket out_data(AVP_TENSOR, timeTick);
            addTick();
            infer(in_data, out_data);
            outStreams[0]->loadPacket(out_data);
            inStreams[0]->releasePacket();
        }
    }
    void processAsync()
    {
        checkStream();
        // @TODO: disentangle while loop in the future!
        bool first_loop = true;
        Stream::iterator next_itr;
        while(true)
        {
            // To get initial iterators
            if(first_loop)
            {
                if(inStreams[0]->empty())
                    continue;
                else
                {
                    iterators.push_back(inStreams[0]->begin());
                    first_loop = false;
                }
            }
            // To avoid end iterator
            else if(next_itr==inStreams[0]->end())
            {
                continue;
            }
            else
            {
                iterators[0] = next_itr;
            }
            
            auto in_data = *iterators[0];
            if(in_data.empty())
            {
                inStreams[0]->releasePacket(iterators[0]);
                break;
            }
            if(in_data.timestamp==timeTick)
                continue;
            else
                timeTick = in_data.timestamp;

            StreamPacket out_data(AVP_TENSOR, timeTick);
            addTick();
            infer(in_data, out_data);
            outStreams[0]->loadPacket(out_data);
            next_itr = iterators[0]+1;
            inStreams[0]->releasePacket(iterators[0]);
        }
    }

};

}