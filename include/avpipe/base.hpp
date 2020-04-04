/*!
 * Copyright (c) 2020 by Contributors
 * author Ruofan Liang
 * base data structure for AV-Pipe
 */
#ifndef AVPIPE_BASE_HPP
#define AVPIPE_BASE_HPP

#include <iostream>
#include <string>
#include <vector>

namespace avp {

template<typename T>
class StreamPackage{
public:
    T data;
    int timestamp;
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
    PPType proc_type;
    virtual void Initialize() = 0;
    virtual void Process(StreamPackage<T>& inData) = 0;
    virtual void Stop() = 0;
};

}

#endif