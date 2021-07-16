#include <iostream>
#include <avpipe/base.hpp>
#include <avpipe/cv_compute.hpp>
// #include <avpipe/tensor_compute.hpp>
#include <thread>

int main()
{
    avp::Stream pipe1, pipe2;
    avp::WebCamProcessor source;
    avp::StreamShowProcessor sink;
    avp::CenterCropResize trans(source.rawHeight, source.rawWidth, 256, 256);
    source.bindStream({}, {&pipe1});
    // trans.bindStream(&pipe1, avp::AVP_STREAM_IN);
    // trans.bindStream(&pipe2, avp::AVP_STREAM_OUT);
    sink.bindStream({&pipe1}, {});
    std::thread srcThreading([&](){
        while(1)
        {
            source.process();
        }
    });
    while(1)
    {
        // trans.process();
        // std::cout<<pipe1.front().mat().size()<<std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        sink.process();
    }
}