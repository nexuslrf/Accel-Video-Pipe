/*!
 * Deprecated
 * The default tensor type is torch::tensor
 */

#pragma once

#include <torch/script.h>
#include "base.hpp"

namespace avp {

using Tensor = torch::Tensor;

class TensorPackage: public StreamPackage {
public:
    Tensor data;
    TensorPackage(Tensor& tensor_data, int tensor_timestamp=-1)
    {
        data = tensor_data;
        timestamp = tensor_timestamp;
    }
    TensorPackage() { timestamp = -1;}
    void* data_ptr()
    {
        return data.data_ptr();
    }
};
}