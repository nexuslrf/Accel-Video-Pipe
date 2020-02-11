#include <fstream>
#include <sstream>
#include <cmath>
#include <functional> 
#include <chrono>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <inference_engine.hpp>
// #ifdef CV_CXX11
#include <mutex>
#include <thread>
#include <queue>
// #endif

using namespace std;

inline void updInput(vector<torch::jit::IValue>& inputs, torch::Tensor& inTensor)
{
    inputs.pop_back();
    inputs.push_back(inTensor);
}

template<typename T>
void fillBlobRandom(InferenceEngine::Blob::Ptr& inputBlob) {
    InferenceEngine::MemoryBlob::Ptr minput = InferenceEngine::as<InferenceEngine::MemoryBlob>(inputBlob);
    if (!minput) {
        THROW_IE_EXCEPTION << "We expect inputBlob to be inherited from MemoryBlob in fillBlobRandom, "
            << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto minputHolder = minput->wmap();

    auto inputBlobData = minputHolder.as<T *>();
    for (size_t i = 0; i < inputBlob->size(); i++) {
        inputBlobData[i] = (T) rand() / RAND_MAX * 10;
    }
}


int modelWidth=192, modelHeight=256;
int batchSize = 1, numJoints=17;
int numLoop = 10;
int main()
{
    torch::NoGradGuard no_grad; // Disable back-grad buffering
    torch::Tensor inTensor, rawOut;
    vector<torch::jit::IValue> inputs;
    inputs.push_back(inTensor);
    torch::jit::script::Module model;
    model = torch::jit::load("../../HRNet-Human-Pose-Estimation/torchScript_pose_resnet_34_256x192.zip");
    for(int i=0; i<numLoop; i++)
    {
        auto stop1 = chrono::high_resolution_clock::now(); 
        inTensor = torch::randn({batchSize, 3, modelHeight, modelWidth}, torch::kFloat32);
        updInput(inputs, inTensor);
        rawOut = model.forward(inputs).toTensor();
        auto stop2 = chrono::high_resolution_clock::now(); 
        cout<<"LibTorch Processing Time: "<<chrono::duration_cast<chrono::milliseconds>(stop2-stop1).count()<<"ms\n";
    }

    string model_xml = "/Users/liangruofan1/Program/HRNet-Human-Pose-Estimation/pose_resnet_34_256x192.xml";
    string model_bin = "/Users/liangruofan1/Program/HRNet-Human-Pose-Estimation/pose_resnet_34_256x192.bin";
    InferenceEngine::Core ie;
    auto network = ie.ReadNetwork(model_xml, model_bin);
    network.setBatchSize(batchSize);
    auto input_info = network.getInputsInfo().begin()->second;
    string input_name = network.getInputsInfo().begin()->first;
    auto output_info = network.getOutputsInfo().begin()->second;
    string output_name = network.getOutputsInfo().begin()->first;
    auto executable_network = ie.LoadNetwork(network, "CPU"); // Unknown problems for GPU version
    auto infer_request = executable_network.CreateInferRequest();
    auto tDesc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32,
                                      {(uint)batchSize, 3, (uint)modelHeight, (uint)modelWidth},
                                      InferenceEngine::Layout::NCHW);
    for(int i=0; i<numLoop; i++)
    {                                  
        auto stop1 = chrono::high_resolution_clock::now();                                
        auto inData = torch::randn({batchSize,3,modelHeight,modelWidth}, torch::kF32);
        InferenceEngine::Blob::Ptr inBlob = InferenceEngine::make_shared_blob<float_t>(tDesc, (float_t*)inData.data_ptr());
        infer_request.SetBlob(input_name, inBlob);
        infer_request.Infer();
        InferenceEngine::Blob::Ptr output = infer_request.GetBlob(output_name);
        auto stop2 = chrono::high_resolution_clock::now(); 
        cout<<"OpenVINO Processing Time: "<<chrono::duration_cast<chrono::milliseconds>(stop2-stop1).count()<<"ms\n";
    }
}