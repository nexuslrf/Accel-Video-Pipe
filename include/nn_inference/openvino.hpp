/*!
 * Note current openvino interface only consider CPU device
 */
#pragma once

#include <inference_engine.hpp>
#include "nn_inference.hpp"

namespace avp {

template<typename T>
class OpenVinoProcessor: public NNProcessor<T> {
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::InferRequest inferRequest;
    InferenceEngine::TensorDesc tDesc;
    InferenceEngine::ExecutableNetwork executableNN;
    string inputName, outputName;
public:
    OpenVinoProcessor(SizeVector dims, DataLayout data_layout, std::string model_path, 
        std::string pp_name = ""): NNProcessor(dims, OPENVINO, data_layout, pp_name)  
    {
        string model_xml = model_path+".xml";
        string model_bin = model_path+".bin";
        network = ie.ReadNetwork(model_xml, model_bin);
        network.setBatchSize(batchSize);
        inputName = network.getInputsInfo().begin()->first;
        outputName = network.getOutputsInfo().begin()->first;
        executableNN = ie.LoadNetwork(network, "CPU");
        inferRequest = executableNN.CreateInferRequest();
        tDesc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, dims,
                                        InferenceEngine::Layout::NCHW);
    } 
    void Process(StreamPackage<T>& inData, StreamPackage<T>& outData)
    {
        InferenceEngine::Blob::Ptr inBlob = InferenceEngine::make_shared_blob<float_t>(tDesc, 
            (float_t*)inData.data.data_ptr());
        inferRequest.SetBlob(inputName, inBlob);
        infer_request.Infer();
        InferenceEngine::Blob::Ptr output = infer_request.GetBlob(outputName);
        
    }

};

}