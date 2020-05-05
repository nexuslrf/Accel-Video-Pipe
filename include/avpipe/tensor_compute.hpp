#pragma once

#include "../tensor_compute/libtorch.hpp"
#include "../tensor_compute/nn_inference.hpp"
#include "../tensor_compute/onnx_runtime.hpp"
#include "../tensor_compute/openvino.hpp"

#ifdef _TENSORRT
#include "../tensor_compute/tensorrt.hpp"
#endif

#include "../tensor_compute/tensor_utils.hpp"