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
#include <onnxruntime_cxx_api.h>
// #ifdef CV_CXX11
#include <mutex>
#include <thread>
#include <queue>
// #endif

using namespace std;

int modelWidth = 256, modelHeight = 256;
int batchSize = 1;
int numKeypoints = 21;

int main()
{
    // file vars
    string rootDir = "/Users/liangruofan1/Program/CV_Models/hand_keypoint_3d/";
    string handModel = rootDir + "blaze_hand.onnx";
    string testImg = rootDir + "pics/test7.jpg";

    // opencv vars
    int rawHeight, rawWidth;
    cv::Mat rawFrame, tmpFrame, inputFrame, resizeFrame;

    // libtorch vars
    torch::NoGradGuard no_grad; // Disable back-grad buffering
    torch::Tensor inputRawTensor, inputTensor, rawKeypts, rawScores;

    /* ---- init ONNX rt ---- */
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, handModel.c_str(), session_options);
    
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t> input_node_dims = {batchSize, 3, modelHeight, modelWidth};
    size_t input_tensor_size = batchSize * 3 * modelHeight * modelWidth;
    std::vector<float> input_tensor_values(input_tensor_size);
    std::vector<const char*> input_node_names = {"input"};
    std::vector<const char*> output_node_names = {"output1", "output2"};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    /* ---- init opencv src file ---- */
    rawFrame = cv::imread(testImg);
    rawHeight = rawFrame.rows;
    rawWidth = rawFrame.cols;
    resize(rawFrame, resizeFrame, cv::Size(modelWidth, modelHeight), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resizeFrame, tmpFrame, cv::COLOR_BGR2RGB);
    tmpFrame.convertTo(inputFrame, CV_32F);
    inputFrame = inputFrame / 127.5 - 1.0;

    /* ---- NN Inference ---- */
    inputRawTensor = torch::from_blob(inputFrame.data, {modelHeight, modelWidth, 3});
    inputTensor = inputRawTensor.permute({2,0,1}).to(torch::kCPU, false, true);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
                (float_t*)inputTensor.data_ptr(), input_tensor_size, input_node_dims.data(), 4);
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
    // cout<<output_tensors.size()<<"\n";
    float* rawKeyptsPtr = output_tensors[0].GetTensorMutableData<float>();
    float* rawScoresPtr = output_tensors[1].GetTensorMutableData<float>();
    // for(int i=0; i<10; i++)
    //     cout<<rawBox[i]<<"  ";
    // Decode Box
    rawKeypts = torch::from_blob(rawKeyptsPtr, {batchSize, numKeypoints, 3});
    rawScores = torch::from_blob(rawScoresPtr, {batchSize, }); 

    cout<<rawScores<<endl;
    /* ---- Show Res ---- */
    auto det = rawKeypts.accessor<float, 3>();
    for(int i=0; i<numKeypoints; i++)
    {
        auto kp_x = det[0][i][0] * rawWidth / modelWidth;
        auto kp_y = det[0][i][1] * rawHeight / modelHeight;
        cv::circle(rawFrame, cv::Point2f(kp_x, kp_y), 2, {0, 255, 0});
    }
    cv::imshow("HandLandmark", rawFrame);
    cv::waitKey();
}