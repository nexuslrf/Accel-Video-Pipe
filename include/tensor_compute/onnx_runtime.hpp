#pragma once

#include <onnxruntime_cxx_api.h>
#include "nn_inference.hpp"

namespace avp {

class ONNXRuntimeProcessor: public NNProcessor {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
    Ort::SessionOptions sessionOptions;
    Ort::Session* sessionPtr;
    Ort::MemoryInfo memInfo;
    size_t inputTensorSize;
    std::vector<int64_t> inDims;
    size_t numInputNodes, numOutputNodes;
    std::vector<const char*> inputNodeNames, outputNodeNames;
public:
    ONNXRuntimeProcessor(SizeVector dims, DataLayout data_layout, std::string model_path, int num_output=1,
        std::string pp_name = ""): NNProcessor(dims, ONNX_RT, data_layout, num_output, pp_name), 
        memInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) 
    {
        inDims = std::vector<int64_t>({(int64_t)this->batchSize, (int64_t)this->channels, 
                (int64_t)this->inHeight, (int64_t)this->inWidth});
        inputTensorSize = this->batchSize * this->channels * this->inHeight * this->inWidth;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        const char* modelPath = model_path.c_str();
        sessionPtr = new Ort::Session(env, modelPath, sessionOptions);
        Ort::AllocatorWithDefaultOptions allocator;
        numInputNodes = sessionPtr->GetInputCount();
        
        inputNodeNames = std::vector<const char*>(numInputNodes);
        // Note: ensure numInputNodes == 1
        char* input_name = sessionPtr->GetInputName(0, allocator);
        inputNodeNames[0] = input_name;
        Ort::TypeInfo typeInfo = sessionPtr->GetInputTypeInfo(0);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

        numOutputNodes = sessionPtr->GetOutputCount();
        outputNodeNames = std::vector<const char*>(numOutputNodes);
        for (size_t i = 0; i < numOutputNodes; i++) {
            char* output_name = sessionPtr->GetOutputName(i, allocator);
            outputNodeNames[i] = output_name;
        }
    }
    ~ONNXRuntimeProcessor()
    {
        delete sessionPtr;
    }
    void infer(StreamPacket& in_data, StreamPacket& out_data)
    {
        Ort::Value inTensor = Ort::Value::CreateTensor<float>(memInfo, (float_t*)in_data.data_ptr(), 
            inputTensorSize, inDims.data(), inDims.size());
        auto outputTensors = sessionPtr->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), 
                &inTensor, numInputNodes, outputNodeNames.data(), numOutputNodes);
        float* rawPtr = outputTensors[0].Ort::Value::template GetTensorMutableData<float>();
        auto outDims = outputTensors[0].GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
        out_data.tensor = torch::from_blob(rawPtr, {outDims[0], outDims[1], outDims[2], outDims[3]});
    }
};

}