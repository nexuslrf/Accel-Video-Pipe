#include <iostream>
#include <avpipe/base.hpp>
#include <cv_compute/cv_stream_src.hpp>
#include <cv_compute/cv_display.hpp>

int main()
{
    avp::Stream nexus;
    avp::WebCamProcessor source;
    avp::StreamShowProcessor sink;
    source.BindStream(&nexus, avp::AVP_STREAM_OUT);
    sink.BindStream(&nexus, avp::AVP_STREAM_IN);
    while(1)
    {
        source.Process();
        sink.Process();
    }
}