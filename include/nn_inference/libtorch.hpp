#pragma once

#include <torch/script.h>
#include "nn_inference.hpp"

namespace avp {

template<typename T>
class LibTorchProcessor: public NNProcessor<T> {
    torch::jit::script::Module model;
    std::vector<torch::jit::IValue> inputs;
public:
    LibTorchProcessor(SizeVector dims, DataLayout data_layout, std::string model_path, 
        std::string pp_name = ""): NNProcessor<T>(dims, LIBTORCH, data_layout, pp_name)    
    {
        torch::NoGradGuard no_grad;
        model = torch::jit::load(model_path);
    }
    void Process(StreamPackage<T>& in_data, StreamPackage<T>& out_data)
    {
        torch::NoGradGuard no_grad;
        inputs.push_back(in_data.data);
        out_data.data = model.forward(inputs).toTensor();
        inputs.pop_back();
    }
};
}