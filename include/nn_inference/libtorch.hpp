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
    void Infer(StreamPacket& in_data, StreamPacket& out_data)
    {
        torch::NoGradGuard no_grad;
        inputs.push_back(in_data.tensor);
        out_data.tensor = model.forward(inputs).toTensor();
        inputs.pop_back();
    }
};
}