/*!
 * Simple warping for TensorRT Inference Engine
 */
#pragma once

#include "tensorrt_util/buffers.h"
#include "tensorrt_util/common.h"
#include "tensorrt_util/logger.h"
#include "tensorrt_util/logging.h"
#include "tensorrt_util/parserOnnxConfig.h"

#include "NvInfer.h"

#include <cuda_runtime_api.h>
#include "nn_inference.hpp"

namespace avp {

/*
 * Note: Right now, only support ONNX format. Other NN formats may be supported in the future.
 * Serilization is also a TODO option.
 */
class TensorRTProcessor: public NNProcessor {
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
    Logger gLogger{Logger::Severity::kINFO};
    int dlaCore{-1};
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::vector<std::string> inputNames, outputNames;
    SampleUniquePtr<nvinfer1::IExecutionContext> context;
    samplesCommon::BufferManager buffers;
    std::vector<void*> hostDataBuffers;
    std::vector<std::vector<int64_t>> mOutputDims;
public:
    TensorRTProcessor(SizeVector input_dims, DataLayout data_layout, std::string model_path, long long workspace_size_MiB=2048, int num_output=1,
        std::string pp_name = "TensorRTProcessor"): NNProcessor(input_dims, TENSORRT, data_layout, num_output, pp_name)
    {
        auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
        auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
        auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));

        auto parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
        // checking ?
        builder->setMaxBatchSize(batchSize);
        config->setMaxWorkspaceSize(workspace_size_MiB * (1<<20));
        samplesCommon::enableDLA(builder.get(), config.get(), dlaCore);
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
        buffers.initBufferManager(mEngine, batchSize);
        context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        auto numInput = network->getNbInputs();
        auto numOutput = network->getNbOutputs();
        for(int i=0; i < numInput; i++)
        {
            inputNames.push_back(network->getInput(i)->getName());
            hostDataBuffers.push_back(static_cast<void*>(buffers.getHostBuffer(inputNames[i])));
        }
        for(int i=0; i < numOutput; i++)
        {
            outputNames.push_back(network->getOutput(i)->getName());
            auto dims = network->getOutput(i)->getDimensions();
            mOutputDims.push_back(std::vector<int64_t>(dims.d, dims.d + dims.nbDims));
        }
    }
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        int bs = in_data_list[0].tensor().size(0);
        for(size_t i=0;  i < numInStreams; i++)
        {
            auto inTensor = in_data_list[i].tensor();
            memcpy(hostDataBuffers[i], inTensor.data_ptr(), inTensor.numel()*sizeof(float));
        }
        buffers.copyInputToDevice();
        bool status = context->execute(bs, buffers.getDeviceBindings().data());
        buffers.copyOutputToHost();
        for(size_t i=0; i < numOutStreams; i++)
        {
            void* outPtr = static_cast<void*>(buffers.getHostBuffer(outputNames[i]));
            std::vector<int64_t> dims({bs});
            dims.insert(dims.end(), mOutputDims[i].begin(), mOutputDims[i].end());
            Tensor output;
            if(avp::numThreads > 1)
            {
                output = torch::empty(dims, torch::kF32);
                memcpy(output.data_ptr(), outPtr, output.numel() * sizeof(float));
            }
            else
                output = torch::from_blob(outPtr, dims, torch::kF32);
            out_data_list[i].loadData(output);
        }
    }
};
}