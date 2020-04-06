/*!
 * Copyright (c) 2020 by Contributors
 * author Ruofan Liang
 * base data structure for AV-Pipe
 * Suppose default stream data type is torch::tensor
 */
#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace avp {

using SizeVector = std::vector<size_t>;

template<typename T>
class StreamPackage{
public:
    T data;
    int timestamp;
    virtual void* data_ptr()
    {
        return NULL;
    }
};

/*! pipe processor type */
enum PPType {
    STREAM_PROC = 0,
    STREAM_INIT = 1,
    STREAM_SINK = 2
};

template<typename T>
class PipeProcessor{
public:
    std::string name;
    PPType procType;
    PipeProcessor(std::string pp_name, PPType pp_type): name(pp_name), procType(pp_type)
    {}
    virtual void Process(StreamPackage<T>& in_data, StreamPackage<T>& out_data) = 0;
    // virtual void Stop() = 0;
};
}
