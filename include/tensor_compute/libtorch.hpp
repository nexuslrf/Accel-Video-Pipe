#pragma once

#include <torch/script.h>
#include "nn_inference.hpp"

namespace avp {

class LibTorchProcessor: public NNProcessor {
    torch::jit::script::Module model;
    std::vector<torch::jit::IValue> inputs;
public:
    LibTorchProcessor(SizeVector dims, DataLayout data_layout, std::string model_path, int num_output=1,
        std::string pp_name = "LibTorchProcessor"): NNProcessor(dims, LIBTORCH, data_layout, num_output, pp_name)    
    {
        torch::NoGradGuard no_grad;
        model = torch::jit::load(model_path);
    }
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        torch::NoGradGuard no_grad;
        inputs.push_back(in_data_list[0].tensor());
        if(numOutStreams==1)
        {
            auto output = model.forward(inputs).toTensor();
            out_data_list[0].loadData(output);
        }
        else //if(numOutStreams>1)
        {
            auto outputs = model.forward(inputs).toTuple();
            for(size_t i=0; i<numOutStreams; i++)
            {
                auto output = outputs->elements()[0].toTensor();
                out_data_list[i].loadData(output);
            }
        }
        inputs.pop_back();
    }
};
}