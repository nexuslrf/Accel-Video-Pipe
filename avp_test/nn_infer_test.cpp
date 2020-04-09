#include <iostream>
#include <nn_inference/libtorch.hpp>
#include <nn_inference/openvino.hpp>
#include <nn_inference/onnx_runtime.hpp>
#include <avpipe/base.hpp>

int main()
{
    avp::SizeVector dims({2,3,256,192});
    auto inTensor = torch::randn({2,3,256,192}, torch::kFloat32);
    avp::StreamPacket inData(inTensor, 0), endPacket;
    std::string model_name = "/Users/liangruofan1/Program/CV_Models/HRNet-Human-Pose-Estimation/pose_resnet_34_256x192";
    /* LibTorch Interface */
    {
        avp::Stream inStream, outStream;
        std::cout<<"============\nLibTorch Test:\n";
        std::string model_path = model_name+".zip";
        avp::LibTorchProcessor CNN(dims, avp::NCHW, model_path);
        std::cout<<"Bind data stream\n";
        CNN.BindStream(&inStream, avp::AVP_STREAM_IN);
        CNN.BindStream(&outStream, avp::AVP_STREAM_OUT);
        std::cout<<"Load data packet\n";
        inStream.LoadPacket(inData);
        std::cout<<"Start Processing\n";
        CNN.Process();
        std::cout<<"Finish Processing\n    Output Sizes:";
        auto outData = outStream.front();
        std::cout<<outData.tensor.sizes()<<std::endl;
        std::cout<<"    Sum of Output: "<<outData.tensor.sum()<<std::endl;
    }
    /* OpenVINO CPU */
    {
        avp::Stream inStream, outStream;
        std::cout<<"============\nOpenVINO CPU Test:\n";
        std::string model_path = model_name;
        avp::OpenVinoProcessor CNN(dims, avp::NCHW, model_path);
        std::cout<<"Bind data stream\n";
        CNN.BindStream(&inStream, avp::AVP_STREAM_IN);
        CNN.BindStream(&outStream, avp::AVP_STREAM_OUT);
        std::cout<<"Load data packet\n";
        inStream.LoadPacket(inData);
        std::cout<<"Start Processing\n";
        CNN.Process();
        std::cout<<"Finish Processing\n    Output Sizes:";
        auto outData = outStream.front();
        std::cout<<outData.tensor.sizes()<<std::endl;
        std::cout<<"    Sum of Output: "<<outData.tensor.sum()<<std::endl;
    }
    /* ONNX Runtime */
    {
        avp::Stream inStream, outStream;
        std::cout<<"============\nONNX Runtime Test:\n";
        std::string model_path = model_name+".onnx";
        avp::ONNXRuntimeProcessor CNN(dims, avp::NCHW, model_path);
        std::cout<<"Bind data stream\n";
        CNN.BindStream(&inStream, avp::AVP_STREAM_IN);
        CNN.BindStream(&outStream, avp::AVP_STREAM_OUT);
        std::cout<<"Load data packet\n";
        inStream.LoadPacket(inData);
        std::cout<<"Start Processing\n";
        CNN.Process();
        std::cout<<"Finish Processing\n    Output Sizes:";
        auto outData = outStream.front();
        std::cout<<outData.tensor.sizes()<<std::endl;
        std::cout<<"    Sum of Output: "<<outData.tensor.sum()<<std::endl;
    }
}