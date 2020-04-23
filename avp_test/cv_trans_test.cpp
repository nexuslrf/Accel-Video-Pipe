#include <iostream>
#include <avpipe/base.hpp>
#include <avpipe/cv_compute.hpp>
#include <avpipe/tensor_compute.hpp>

int main()
{
    if(0){
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
            std::cout<<pipe1.front().mat().size()<<std::endl;
            sink.process();
        }
    }

    if(1){
        avp::DataList in_data_list, out_data_list;
        avp::StreamPacket inData(avp::AVP_MAT), outData;
        inData.loadData(avp::Mat(4,5,CV_32FC3,{1,2,3}));
        inData.loadData(avp::Mat(4,5,CV_32FC3,{3,2,1}));
        in_data_list.push_back(inData);
        out_data_list.push_back(outData);
        avp::DataLayoutConvertion toTensor;
        toTensor.run(in_data_list, out_data_list);
        std::cout<<out_data_list[0].tensor().sizes()<<"\n";
    }
}