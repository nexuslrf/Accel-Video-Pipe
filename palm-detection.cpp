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
int numAnchors = 2944, batchSize = 1;

cv::Mat cropResize(const cv::Mat& frame, int xMin, int yMin, int xCrop, int yCrop);
inline void matToTensor(cv::Mat const& src, torch::Tensor& out);

int main()
{
    string rootDir = "/Users/liangruofan1/Program/CV_Models/palm_detector/";
    string anchorFile = rootDir + "anchors.bin";
    string palmModel = rootDir + "palm_detection.onnx";
    string testImg = rootDir + "pics/palm_test2.jpg";

    fstream fin(anchorFile, ios::in | ios::binary);
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto anchors = torch::empty({numAnchors, 4}, options);
    // load anchor binary file
    fin.read((char *)anchors.data_ptr(), anchors.numel() * sizeof(float));
    // [Debug] cout<<anchors.slice(0, 0, 6);
    // load ONNX model
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, palmModel.c_str(), session_options);
    auto frame = cv::imread(testImg);
    int rawHeight = frame.rows, rawWidth = frame.cols;
    int cropWidthLowBnd, cropWidth, cropHeightLowBnd, cropHeight;

    torch::Tensor inputRawTensor = torch::empty({modelHeight, modelWidth, 3}, torch::kFloat32),
                inputTensor;

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t> input_node_dims = {batchSize, 3, modelHeight, modelWidth};
    size_t input_tensor_size = batchSize * 3 * modelHeight * modelWidth;
    std::vector<float> input_tensor_values(input_tensor_size);
    std::vector<const char*> input_node_names = {"input"};
    std::vector<const char*> output_node_names = {"output1", "output2"};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // show pic
    // cv::imshow("PalmDetection", frame);
    // cv::waitKey();
    // OpenCV pre-processing
    cv::Mat showFrame, cropFrame, tmpFrame;
    // cropping long edge
    if(rawHeight > rawWidth)
    {
        cropHeightLowBnd = (rawHeight - rawWidth) / 2;
        cropWidthLowBnd = 0;
        cropHeight = cropWidth = rawWidth;
    }
    else
    {
        cropWidthLowBnd = (rawWidth - rawHeight) / 2;
        cropHeightLowBnd = 0;
        cropHeight = cropWidth = rawHeight;
    }
    showFrame = cropResize(frame, cropWidthLowBnd, cropHeightLowBnd, cropWidth, cropHeight);
    cv::cvtColor(showFrame, tmpFrame, cv::COLOR_BGR2RGB);
    tmpFrame.convertTo(cropFrame, CV_32F);
    cropFrame = cropFrame / 127.5 - 1.0;
    cout<<cropFrame({0,0,4,4})<<endl;
    matToTensor(cropFrame, inputRawTensor);
    inputTensor = inputRawTensor.permute({2,0,1}).to(torch::kCPU, false, true);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
                (float_t*)inputTensor.data_ptr(), input_tensor_size, input_node_dims.data(), 4);
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
    cout<<output_tensors.size()<<"\n";
    float* rawBox = output_tensors[0].GetTensorMutableData<float>();
    float* rawScore = output_tensors[1].GetTensorMutableData<float>();
    // Decode Box

}

cv::Mat cropResize(const cv::Mat& frame, int xMin, int yMin, int xCrop, int yCrop)
{
    cv::Mat tempImg;
    // crop and resize
    cv::Rect ROI(xMin, yMin, xCrop, yCrop);
    resize(frame(ROI), tempImg, cv::Size(modelWidth, modelHeight), 0, 0, cv::INTER_LINEAR);
    // normalize

    return tempImg; 
}

inline void matToTensor(cv::Mat const& src, torch::Tensor& out)
{
    // stride=width*channels
    memcpy(out.data_ptr(), src.data, out.numel() * sizeof(float));
    return;
}

void decodeBoxes(const torch::Tensor &rawBoxes, const torch::Tensor &anchors, torch::Tensor &boxes)
{
    float x_center
}