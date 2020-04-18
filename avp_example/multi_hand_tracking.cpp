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
    int numAnchors = 2944, numKeypointsPalm = 7;

    avp::CenterCropResize crop(rawHeight, rawWidth, dstHeight, dstWidth);
    avp::ImgNormalization normalization(0.5, 0.5);
    avp::DataLayoutConvertion matToTensor;
    avp::ONNXRuntimeProcessor CNN({1,3,256,256}, avp::NCHW, palmModel, 2);
    avp::DecodeDetBoxes decodeBoxes(numAnchors, anchorFile, dstHeight, dstWidth, numKeypointsPalm);
    avp::NonMaxSuppression NMS(numKeypointsPalm);
    avp::DrawDetBoxes drawDet(dstHeight, dstWidth);
    avp::StreamShowProcessor imshow(-1);

    avp::Stream pipe[15];

    crop.bindStream(&pipe[0], avp::AVP_STREAM_IN);
    crop.bindStream(&pipe[1], avp::AVP_STREAM_OUT);
    normalization.bindStream(&pipe[1], avp::AVP_STREAM_IN);
    normalization.bindStream(&pipe[2], avp::AVP_STREAM_OUT);
    matToTensor.bindStream(&pipe[2], avp::AVP_STREAM_IN);
    matToTensor.bindStream(&pipe[3], avp::AVP_STREAM_OUT);
    CNN.bindStream(&pipe[3], avp::AVP_STREAM_IN);
    CNN.bindStream(&pipe[4], avp::AVP_STREAM_OUT);
    CNN.bindStream(&pipe[5], avp::AVP_STREAM_OUT);
    decodeBoxes.bindStream(&pipe[4], avp::AVP_STREAM_IN);
    decodeBoxes.bindStream(&pipe[6], avp::AVP_STREAM_OUT);
    NMS.bindStream(&pipe[5], avp::AVP_STREAM_IN);
    NMS.bindStream(&pipe[6], avp::AVP_STREAM_IN);
    NMS.bindStream(&pipe[7], avp::AVP_STREAM_OUT);
    NMS.bindStream(&pipe[8], avp::AVP_STREAM_OUT);
    drawDet.bindStream(&pipe[7], avp::AVP_STREAM_IN);
    drawDet.bindStream(&pipe[1], avp::AVP_STREAM_IN);
    drawDet.bindStream(&pipe[10], avp::AVP_STREAM_OUT);
    imshow.bindStream(&pipe[10], avp::AVP_STREAM_IN);

    avp::StreamPacket inData(rawFrame, 0);
    pipe[0].loadPacket(inData);

    crop.process();
    std::cout<<"crop pass!\n";
    normalization.process();
    std::cout<<"normalization pass!\n";
    matToTensor.process();
    std::cout<<"matToTensor pass!\n";
    CNN.process();
    std::cout<<"CNN pass!\n";
    decodeBoxes.process();
    std::cout<<"decodeBoxes pass!\n";
    NMS.process();
    std::cout<<"NMS pass!\n";
    // std::cout<<pipe[7].front().tensor.sizes()<<"\n";
    drawDet.process();
    std::cout<<"drawDet pass!\n";
    imshow.process();
    std::cout<<"imshow pass!\n";


}