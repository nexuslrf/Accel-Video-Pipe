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
int numAnchors = 2944;

cv::Mat cropResize(const cv::Mat& frame, int xMin, int yMin, int xCrop, int yCrop);

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
    // show pic
    cv::imshow("PalmDetection", frame);
    cv::waitKey();
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
