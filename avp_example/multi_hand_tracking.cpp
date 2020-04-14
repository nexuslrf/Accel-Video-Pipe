#include <iostream>
#include <avpipe/tensor_compute.hpp>
#include <avpipe/cv_compute.hpp>
#include <avpipe/base.hpp>
#include <tensor_compute/det_postprocess.hpp>

int main()
{
    std::string rootDir = "/Users/liangruofan1/Program/CV_Models/";
    std::string anchorFile = rootDir + "palm_detector/anchors.bin";
    std::string palmModel = rootDir + "palm_detector/palm_detection.onnx";
    std::string handModel = rootDir + "hand_keypoint_3d/blaze_hand.onnx";
    std::string testImg = rootDir + "palm_detector/pics/LRF.jpg";

    auto rawFrame = cv::imread(testImg);
    int rawHeight = rawFrame.rows;
    int rawWidth = rawFrame.cols;
    int dstHeight = 256, dstWidth = 256;

    avp::CenterCropResize crop(rawHeight, rawWidth, dstHeight, dstWidth);
    avp::ImgNormalization normalization(0.5, 0.5);
    avp::DataLayoutConvertion matToTensor;
    avp::ONNXRuntimeProcessor CNN({1,3,256,256}, avp::NCHW, palmModel);
    
}