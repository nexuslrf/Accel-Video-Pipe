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

};

}