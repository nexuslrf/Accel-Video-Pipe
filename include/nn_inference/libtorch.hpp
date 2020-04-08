#pragma once

#include <torch/script.h>
#include "nn_inference.hpp"

namespace avp {

class LibTorchProcessor: public NNProcessor {
    torch::jit::script::Module model;
    std::vector<torch::jit::IValue> inputs;
public:
    LibTorchProcessor(SizeVector dims, DataLayout data_layout, std::string model_path, 
        std::string pp_name = ""): NNProcessor(dims, LIBTORCH, data_layout, pp_name)    
    {
        torch::NoGradGuard no_grad;
        model = torch::jit::load(model_path);
    }
    void Process()
    {
        torch::NoGradGuard no_grad;
        if(inStreams.empty()||outStreams.empty())
        {
            std::cerr<<"inStreams are empty!\n";
            exit(0);
        }
        while(!inStreams[0]->empty()){
            auto in_data = inStreams[0]->front();
            StreamPackage out_data;
            inputs.push_back(in_data.tensor);
            out_data.tensor = model.forward(inputs).toTensor();
            outStreams[0]->push(out_data);
            inputs.pop_back();
            inStreams[0]->Consume();
        }
    }
};
}