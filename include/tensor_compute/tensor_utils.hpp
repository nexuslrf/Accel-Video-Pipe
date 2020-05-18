#pragma once

#include "../avpipe/base.hpp"
#include "nn_inference.hpp"

namespace avp {
/*!
 * Do the mat to tensor convertion
 * HWC to CHW convertion
 * gather multiple packet to form a batch...
 */
class DataLayoutConvertion: public PipeProcessor {
public:
    DataLayout inLayout, outLayout;
    DataLayoutConvertion(DataLayout in_layout=NHWC, DataLayout out_layout=NCHW, std::string pp_name="DataLayoutConvertion"): 
        PipeProcessor(1,1, AVP_TENSOR, pp_name, STREAM_PROC),
        inLayout(in_layout), outLayout(out_layout)
    {}
    // Right now only consider HWC to CHW...
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        int bs = in_data_list[0].size();
        auto size = in_data_list[0].mat().size();
        auto objTensor = torch::empty({bs, 3, size.height, size.width}, torch::kF32);
        for(int i=0; i<bs; i++)
        {
            auto tmp_out = in_data_list[0].tensor(i).permute({2,0,1});
            objTensor[i] = tmp_out;
            // auto new_out = tmp_out.permute({2,0,1}).unsqueeze(0); //.to(torch::kCPU, false, true);
            // std::cout<<new_out.sizes()<<"\n";
        }
        out_data_list[0].loadData(objTensor);
    }
};

}