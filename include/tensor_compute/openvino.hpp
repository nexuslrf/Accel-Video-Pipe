/*!
 * Note current openvino interface only consider CPU device
 */
#pragma once

#include <inference_engine.hpp>
#include "nn_inference.hpp"

namespace avp {

class OpenVinoProcessor: public NNProcessor {
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::InferRequest inferRequest;
    InferenceEngine::TensorDesc tDesc;
    InferenceEngine::ExecutableNetwork executableNN;
    std::vector<std::string> inputNames, outputNames;
    SizeVector dims;
public:
    OpenVinoProcessor(SizeVector input_dims, DataLayout data_layout, std::string model_path, int num_output=1,
        std::string pp_name = "OpenVinoProcessor"): NNProcessor(input_dims, OPENVINO, data_layout, num_output, pp_name), dims(input_dims)  
    {
        std::string model_xml = model_path+".xml";
        std::string model_bin = model_path+".bin";
        network = ie.ReadNetwork(model_xml, model_bin);
        network.setBatchSize(this->batchSize);
        inputNames.push_back(network.getInputsInfo().begin()->first);
        for(auto &pr: network.getOutputsInfo())
            outputNames.push_back(pr.first);
        executableNN = ie.LoadNetwork(network, "CPU");
        inferRequest = executableNN.CreateInferRequest();
        if(batchSize>0)
        {
            tDesc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, input_dims,
                                        InferenceEngine::Layout::NCHW);
        }
    } 
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        if(batchSize<=0)
        {
            dims[0] = in_data_list[0].tensor().size(0);
            tDesc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, dims,
                                        InferenceEngine::Layout::NCHW);
        }
        InferenceEngine::Blob::Ptr inBlob = InferenceEngine::make_shared_blob<float_t>(tDesc, 
            (float_t*)in_data_list[0].data_ptr());
        inferRequest.SetBlob(inputNames[0], inBlob);
        inferRequest.Infer();
        for(size_t i=0; i<numOutStreams; i++)
        {
            InferenceEngine::Blob::Ptr outBlob = inferRequest.GetBlob(outputNames[i]);
            auto dims_tmp = outBlob->getTensorDesc().getDims();
            std::vector<int64_t> dims(dims_tmp.begin(), dims_tmp.end());
            Tensor output;
            if(avp::numThreads > 1)
            {
                output = torch::empty(dims, torch::kF32);
                memcpy(output.data_ptr(), outBlob->buffer().as<float*>(), output.numel() * sizeof(float));
            }
            else
                output = torch::from_blob(outBlob->buffer().as<float*>(), dims);
            out_data_list[i].loadData(output);
        }
    }
};
}