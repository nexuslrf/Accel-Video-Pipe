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
    int batchSize; 
    DataLayout inLayout, outLayout;
    DataLayoutConvertion(int batch_size=1, DataLayout in_layout=NHWC, DataLayout out_layout=NCHW, std::string pp_name=""): 
        PipeProcessor(1,1, AVP_TENSOR, pp_name, STREAM_PROC), batchSize(batch_size), 
        inLayout(in_layout), outLayout(out_layout)
    {}
    // Right now only consider HWC to CHW...
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        // only if batchsize == 1 ...
        auto output = in_data_list[0].tensor().permute({2,0,1}).unsqueeze(0).to(torch::kCPU, false, true);
        out_data_list[0].loadData(output);
    }
};

}