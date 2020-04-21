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
public:
    OpenVinoProcessor(SizeVector dims, DataLayout data_layout, std::string model_path, int num_output=1,
        std::string pp_name = ""): NNProcessor(dims, OPENVINO, data_layout, num_output, pp_name)  
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
        tDesc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, dims,
                                        InferenceEngine::Layout::NCHW);
    } 
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        InferenceEngine::Blob::Ptr inBlob = InferenceEngine::make_shared_blob<float_t>(tDesc, 
            (float_t*)in_data_list[0].data_ptr());
        inferRequest.SetBlob(inputNames[0], inBlob);
        inferRequest.Infer();
        for(size_t i=0; i<numOutStreams; i++)
        {
            InferenceEngine::Blob::Ptr outBlob = inferRequest.GetBlob(outputNames[i]);
            auto dims_tmp = outBlob->getTensorDesc().getDims();
            std::vector<int64_t> dims(dims_tmp.begin(), dims_tmp.end());
            auto output = torch::from_blob(outBlob->buffer().as<float*>(), dims);
            out_data_list[i].loadData(output);
        }
    }
};
}