/*
Test the compatibility of different Inference Engine with C++11 threading 
*/
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

std::string rootDir = "/Users/liangruofan1/Program/CV_Models/";
std::string anchorFile = rootDir + "palm_detector/anchors.bin";
std::string palmModel = rootDir + "palm_detector/palm_detection.onnx";
std::string handModel = rootDir + "hand_keypoint_3d/blaze_hand.onnx";
std::string testImg = rootDir + "palm_detector/pics/LRF.jpg";

int modelWidth = 256, modelHeight = 256, batchSize = 1;

int main()
{
    /* ONNX Runtime */
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session sessionPalm(env, palmModel.c_str(), session_options);
        Ort::Session sessionHand(env, handModel.c_str(), session_options);
        
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<int64_t> input_node_dims = {batchSize, 3, modelHeight, modelWidth};
        size_t input_tensor_size = batchSize * 3 * modelHeight * modelWidth;
        // std::vector<float> input_tensor_values(input_tensor_size);
        std::vector<const char*> input_node_names = {"input"};
        std::vector<const char*> output_node_names = {"output1", "output2"};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        auto inData = torch::randn({batchSize,3,modelHeight,modelWidth}, torch::kF32);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
                (float_t*)inData.data_ptr(), input_tensor_size, input_node_dims.data(), 4);

        std::thread palmThread([&](){
            for(int i=0; i<1000; i++)
            {
                auto output_tensors = sessionPalm.Run(Ort::RunOptions{nullptr}, 
                    input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
            }
        });

        std::thread handThread([&](){
            for(int i=0; i<1000; i++)
            {
                auto output_tensors = sessionHand.Run(Ort::RunOptions{nullptr}, 
                    input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
            }
        });
        palmThread.join();
        handThread.join();
    }
}