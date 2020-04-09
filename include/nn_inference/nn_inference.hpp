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
    NNProcessor(SizeVector dims, IEType ie_type, DataLayout data_layout,
                std::string pp_name): PipeProcessor(pp_name, STREAM_PROC)
    {
        batchSize = dims[0];
        channels = dims[1];
        inHeight = dims[2];
        inWidth = dims[3];
        ieType = ie_type;
        dataLayout = data_layout;
    }
    virtual void Infer(StreamPacket& in_data, StreamPacket& out_data) = 0;
    void Process()
    {
        if(inStreams.empty()||outStreams.empty())
        {
            std::cerr<<"Streams are empty!\n";
            exit(0);
        }
        // @TODO: disentangle while loop in the future!
        while(!inStreams[0]->empty())
        {
            auto in_data = inStreams[0]->front();
            if(in_data.empty())
                break;
            if(in_data.timestamp==timeTick)
                continue;
            else
                timeTick = in_data.timestamp;
            StreamPacket out_data(AVP_TENSOR, timeTick);
            AddTick();
            Infer(in_data, out_data);
            outStreams[0]->LoadPacket(out_data);
            inStreams[0]->ReleasePacket();
        }
    }

};

}