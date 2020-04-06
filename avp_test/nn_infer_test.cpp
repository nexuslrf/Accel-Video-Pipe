#include <iostream>
#include <nn_inference/libtorch.hpp>
#include <nn_inference/openvino.hpp>
#include <nn_inference/onnx_runtime.hpp>
#include <avpipe/tensor_package.hpp>

int main()
{
    avp::SizeVector dims({2,3,256,192});
    auto inTensor = torch::randn({2,3,256,192}, torch::kFloat32);
    avp::TensorPackage inData(inTensor);
    avp::TensorPackage outData;
    std::string model_name = "/Users/liangruofan1/Program/CV_Models/HRNet-Human-Pose-Estimation/pose_resnet_34_256x192";
    /* LibTorch Interface */
    {
        std::cout<<"============\nLibTorch Test:\n";
        std::string model_path = model_name+".zip";
        avp::LibTorchProcessor<torch::Tensor> CNN(dims, avp::NCHW, model_path);
        std::cout<<"Start Processing\n";
        CNN.Process(inData, outData);
        std::cout<<"Finish Processing\n    Output Sizes:";
        std::cout<<outData.data.sizes()<<std::endl;
        std::cout<<"    Sum of Output: "<<outData.data.sum()<<std::endl;

    }
    /* OpenVINO CPU */
    {
        std::cout<<"============\nOpenVINO CPU Test:\n";
        std::string model_path = model_name;
        avp::OpenVinoProcessor<torch::Tensor> CNN(dims, avp::NCHW, model_path);
        std::cout<<"Start Processing\n";
        CNN.Process(inData, outData);
        std::cout<<"Finish Processing\n    Output Sizes:";
        std::cout<<outData.data.sizes()<<std::endl;
        std::cout<<"    Sum of Output: "<<outData.data.sum()<<std::endl;
    }
    /* ONNX Runtime */
    {
        std::cout<<"============\nONNX Runtime Test:\n";
        std::string model_path = model_name+".onnx";
        avp::ONNXRuntimeProcessor<torch::Tensor> CNN(dims, avp::NCHW, model_path);
        std::cout<<"Start Processing\n";
        CNN.Process(inData, outData);
        std::cout<<"Finish Processing\n    Output Sizes:";
        std::cout<<outData.data.sizes()<<std::endl;
        std::cout<<"    Sum of Output: "<<outData.data.sum()<<std::endl;
    }
}