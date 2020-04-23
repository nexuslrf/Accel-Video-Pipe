#pragma once

#include <onnxruntime_cxx_api.h>
#include "nn_inference.hpp"

namespace avp {

class ONNXRuntimeProcessor: public NNProcessor {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
    Ort::SessionOptions sessionOptions;
    Ort::Session* sessionPtr;
    Ort::MemoryInfo memInfo;
    size_t inputTensorSize, singleTensorSize;
    std::vector<int64_t> inDims;
    size_t numInputNodes, numOutputNodes;
    std::vector<const char*> inputNodeNames, outputNodeNames;
public:
    ONNXRuntimeProcessor(SizeVector dims, DataLayout data_layout, std::string model_path, int num_output=1,
        std::string pp_name = ""): NNProcessor(dims, ONNX_RT, data_layout, num_output, pp_name), 
        memInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) 
    {
        inDims = std::vector<int64_t>(dims.begin(), dims.end());
        singleTensorSize = channels * inHeight * inWidth;
        inputTensorSize = batchSize * singleTensorSize;

        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        const char* modelPath = model_path.c_str();
        sessionPtr = new Ort::Session(env, modelPath, sessionOptions);
        Ort::AllocatorWithDefaultOptions allocator;
        numInputNodes = sessionPtr->GetInputCount();
        
        inputNodeNames = std::vector<const char*>(numInputNodes);
        // Note: ensure numInputNodes == 1
        for(size_t i=0; i<numInputNodes; i++)
        {
            char* input_name = sessionPtr->GetInputName(i, allocator);
            inputNodeNames[i] = input_name;
        }
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
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        if(batchSize<=0)
        {
            int bs = in_data_list[0].tensor().size(0);
            inDims[0] = bs;
            inputTensorSize = singleTensorSize * bs;
        }
        Ort::Value inTensor = Ort::Value::CreateTensor<float>(memInfo, (float_t*)in_data_list[0].data_ptr(), 
            inputTensorSize, inDims.data(), inDims.size());
        auto outputTensors = sessionPtr->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), 
                &inTensor, numInputNodes, outputNodeNames.data(), numOutputNodes);
        for(size_t i=0; i<numOutStreams; i++)
        {
            float* rawPtr = outputTensors[i].Ort::Value::template GetTensorMutableData<float>();
            auto outDims = outputTensors[i].GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
            auto output = torch::from_blob(rawPtr, outDims);
            out_data_list[i].loadData(output);
        }
    }
};

}