#include <iostream>
#include <avpipe/base.hpp>
#include <avpipe/util_compute.hpp>

int main()
{
    avp::Tensor aten1 = torch::arange(12).reshape({3,4});
    avp::Tensor aten2 = torch::arange(16).reshape({4,4});

    avp::Stream pipe[4];
    avp::StreamPacket dataIn1(aten1, 0), dataIn2(aten2, 0);
    
    avp::TemplateProcessor multiplexer(2,2, [](avp::DataList& in_data_list, avp::DataList& out_data_list){
        std::cout<<"Wtf?\n";
        auto t1 = in_data_list[0].tensor();
        auto t2 = in_data_list[1].tensor();
        if(t1.numel()>t2.numel())
        {
            out_data_list[0].loadData(t1);
            out_data_list[1].loadData(t2);
        }
        else
        {
            out_data_list[1].loadData(t1);
            out_data_list[0].loadData(t2);
        }
    });

    multiplexer.bindStream({&pipe[0],&pipe[1]}, avp::AVP_STREAM_IN);
    multiplexer.bindStream({&pipe[2],&pipe[3]}, avp::AVP_STREAM_OUT);
    
    pipe[0].loadPacket(dataIn1);
    pipe[1].loadPacket(dataIn2);

    multiplexer.process();

    std::cout<<pipe[2].front().tensor().sizes()<<std::endl;
    std::cout<<pipe[3].front().tensor().sizes()<<std::endl;
}