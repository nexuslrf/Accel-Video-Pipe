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
    DataLayoutConvertion(DataLayout in_layout=NHWC, DataLayout out_layout=NCHW, std::string pp_name=""): 
        PipeProcessor(1,1, AVP_TENSOR, pp_name, STREAM_PROC),
        inLayout(in_layout), outLayout(out_layout)
    {}
    // Right now only consider HWC to CHW...
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        int bs = in_data_list[0].size();
        if(bs==1)
        {
            auto output = in_data_list[0].tensor().permute({2,0,1}).unsqueeze(0).to(torch::kCPU, false, true);
            out_data_list[0].loadData(output);
        }
        else
        {   
            auto size = in_data_list[0].mat().size();
            auto objTensor = torch::empty({bs, 3, size.height, size.width}, torch::kF32);
            for(int i=0; i<bs; i++)
            {
                auto tmp_out = in_data_list[0].tensor(i).permute({2,0,1}).unsqueeze(0).to(torch::kCPU, false, true);
                objTensor.slice(0, i, i+1) = tmp_out;
            }
            out_data_list[0].loadData(objTensor);
        }
    }
};

}