#pragma once

#include <torch/script.h>
#include "nn_inference.hpp"

namespace avp {

template<typename T>
class LibTorchProcessor: public NNProcessor<T> {
    torch::jit::script::Module model;
public:
    LibTorchProcessor(SizeVector dims, DataLayout data_layout, std::string model_path, 
        std::string pp_name = ""): NNProcessor(dims, LIBTORCH, data_layout, pp_name)    
    {
        torch::NoGradGuard no_grad;
        model = torch::jit::load(model_path);
    }
    void Process(StreamPackage<T>& inData, StreamPackage<T>& outData)
    {
        torch::NoGradGuard no_grad;
        outData.data = model.forward(inData.data).toTensor();
    }
};
}