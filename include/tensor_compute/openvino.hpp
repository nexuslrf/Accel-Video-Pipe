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
    std::string inputName, outputName;
public:
    OpenVinoProcessor(SizeVector dims, DataLayout data_layout, std::string model_path, 
        std::string pp_name = ""): NNProcessor(dims, OPENVINO, data_layout, pp_name)  
    {
        std::string model_xml = model_path+".xml";
        std::string model_bin = model_path+".bin";
        network = ie.ReadNetwork(model_xml, model_bin);
        network.setBatchSize(this->batchSize);
        inputName = network.getInputsInfo().begin()->first;
        outputName = network.getOutputsInfo().begin()->first;
        executableNN = ie.LoadNetwork(network, "CPU");
        inferRequest = executableNN.CreateInferRequest();
        tDesc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, dims,
                                        InferenceEngine::Layout::NCHW);
    } 
    void infer(StreamPacket& in_data, StreamPacket& out_data)
    {
        InferenceEngine::Blob::Ptr inBlob = InferenceEngine::make_shared_blob<float_t>(tDesc, 
            (float_t*)in_data.data_ptr());
        inferRequest.SetBlob(inputName, inBlob);
        inferRequest.Infer();
        InferenceEngine::Blob::Ptr outBlob = inferRequest.GetBlob(outputName);
        auto dims = outBlob->getTensorDesc().getDims();
        int out_batchSize = dims[0], out_channels = dims[1], 
            out_height = dims[2], out_width = dims[3];
        out_data.tensor = torch::from_blob(outBlob->buffer().as<float*>(), 
            {out_batchSize, out_channels, out_height, out_width});
        // outBlob.get();
    }
};
}