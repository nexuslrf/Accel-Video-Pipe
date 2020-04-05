#pragma once

#include <onnxruntime_cxx_api.h>
#include "nn_inference.hpp"

namespace avp {

template<typename T>
class ONNXRuntimeProcessor: public NNProcessor<T> {
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::Session session;
    Ort::MemoryInfo memInfo;
    size_t inputTensorSize;
    SizeVector inDims;
    size_t numInputNodes, numOutputNodes;
    std::vector<const char*> inputNodeNames, outputNodeNames;
public:
    ONNXRuntimeProcessor(SizeVector dims, DataLayout data_layout, std::string model_path, 
        std::string pp_name = ""): NNProcessor(dims, ONNX_RT, data_layout, pp_name) 
    {
        inDims = dims;
        inputTensorSize = batchSize * channels * inHeight * inWidth;
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        const char* modelPath = model_path.c_str();
        session = Ort::Session(env, modelPath, sessionOptions);
        memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::AllocatorWithDefaultOptions allocator;
        numInputNodes = session.GetInputCount();
        inputNodeNames = std::vector(numInputNodes);
        for (int i = 0; i < numInputNodes; i++) {
            char* input_name = session.GetInputName(i, allocator);
            inputNodeNames[i] = input_name;
        }
        numOutputNodes = session.GetOutputCount();
        outputNodeNames = std::vector(numOutputNodes);
        for (int i = 0; i < numOutputNodes; i++) {
            char* output_name = session.GetOutputName(i, allocator);
            outputNodeNames[i] = output_name;
        }
    }
    void Process(StreamPackage<T>& inData, StreamPackage<T>& outData)
    {
        Ort::Value inTensor = Ort::Value::CreateTensor<float>(memInfo, (float_t*)inData.data_ptr(), 
            inputTensorSize, inDims.data(), inDims.size());
        auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), 
                &inTensor, numInputNodes, outputNodeNames.data(), numOutputNodes);
    }
};

}