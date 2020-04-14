#include <iostream>
#include <avpipe/base.hpp>
#include <cv_compute/cv_stream_src.hpp>
#include <cv_compute/cv_display.hpp>
#include <cv_compute/cv_transformation.hpp>

int main()
{
    avp::Stream pipe1, pipe2;
    avp::WebCamProcessor source;
    avp::StreamShowProcessor sink;
    avp::CenterCropResize trans(source.rawHeight, source.rawWidth, 256, 256);
    source.bindStream(&pipe1, avp::AVP_STREAM_OUT);
    // trans.bindStream(&pipe1, avp::AVP_STREAM_IN);
    // trans.bindStream(&pipe2, avp::AVP_STREAM_OUT);
    sink.bindStream(&pipe1, avp::AVP_STREAM_IN);
    // while(1)
    {
        source.process();
        // trans.process();
        std::cout<<pipe1.front().mat.size()<<std::endl;
        sink.process();
    }
}