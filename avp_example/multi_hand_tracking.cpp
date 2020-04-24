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
    int numAnchors = 2944, numKeypointsPalm = 7, numKeypointsHand = 21;

    // std::cout<<rawHeight<<" "<<rawWidth<<"\n";

    avp::CenterCropResize crop(rawHeight, rawWidth, dstHeight, dstWidth, false, true);
    avp::ImgNormalization normalization(0.5, 0.5);
    avp::DataLayoutConvertion matToTensor;
    avp::ONNXRuntimeProcessor PalmCNN({1,3,256,256}, avp::NCHW, palmModel, 2);
    avp::DecodeDetBoxes decodeBoxes(numAnchors, anchorFile, dstHeight, dstWidth, numKeypointsPalm);
    avp::NonMaxSuppression NMS(numKeypointsPalm);
    avp::DrawDetBoxes drawDet(dstHeight, dstWidth);
    avp::StreamShowProcessor imshow(-1);
    avp::RotateCropResize rotateCropResize(dstHeight, dstWidth, crop.cropHeight, crop.cropWidth);
    avp::DataLayoutConvertion multiCropToTensor;
    avp::ImgNormalization normalization2(0.5, 0.5);
    avp::ONNXRuntimeProcessor HandCNN({0,3,256,256}, avp::NCHW, handModel, 2);
    avp::RotateBack rotateBack;
    avp::DrawLandMarks drawKeypoint;
    avp::StreamShowProcessor imshow_kp(-1);
    avp::Stream pipe[25];

    crop.bindStream(&pipe[0], avp::AVP_STREAM_IN);
    crop.bindStream(&pipe[1], avp::AVP_STREAM_OUT);
    crop.bindStream(&pipe[14], avp::AVP_STREAM_OUT);
    normalization.bindStream(&pipe[1], avp::AVP_STREAM_IN);
    normalization.bindStream(&pipe[2], avp::AVP_STREAM_OUT);
    matToTensor.bindStream(&pipe[2], avp::AVP_STREAM_IN);
    matToTensor.bindStream(&pipe[3], avp::AVP_STREAM_OUT);
    PalmCNN.bindStream(&pipe[3], avp::AVP_STREAM_IN);
    PalmCNN.bindStream(&pipe[4], avp::AVP_STREAM_OUT);
    PalmCNN.bindStream(&pipe[5], avp::AVP_STREAM_OUT);
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
    rotateCropResize.bindStream(&pipe[7], avp::AVP_STREAM_IN);
    rotateCropResize.bindStream(&pipe[8], avp::AVP_STREAM_IN);
    rotateCropResize.bindStream(&pipe[14], avp::AVP_STREAM_IN);
    rotateCropResize.bindStream(&pipe[11], avp::AVP_STREAM_OUT);
    rotateCropResize.bindStream(&pipe[12], avp::AVP_STREAM_OUT);
    rotateCropResize.bindStream(&pipe[13], avp::AVP_STREAM_OUT);
    rotateCropResize.bindStream(&pipe[21], avp::AVP_STREAM_OUT);
    normalization2.bindStream(&pipe[11], avp::AVP_STREAM_IN);
    normalization2.bindStream(&pipe[15], avp::AVP_STREAM_OUT);
    multiCropToTensor.bindStream(&pipe[15], avp::AVP_STREAM_IN);
    multiCropToTensor.bindStream(&pipe[16], avp::AVP_STREAM_OUT);
    HandCNN.bindStream(&pipe[16], avp::AVP_STREAM_IN);
    HandCNN.bindStream(&pipe[17], avp::AVP_STREAM_OUT);
    HandCNN.bindStream(&pipe[18], avp::AVP_STREAM_OUT);
    rotateBack.bindStream(&pipe[17], avp::AVP_STREAM_IN);
    rotateBack.bindStream(&pipe[18], avp::AVP_STREAM_IN);
    rotateBack.bindStream(&pipe[21], avp::AVP_STREAM_IN);
    rotateBack.bindStream(&pipe[12], avp::AVP_STREAM_IN);
    rotateBack.bindStream(&pipe[13], avp::AVP_STREAM_IN);
    rotateBack.bindStream(&pipe[19], avp::AVP_STREAM_OUT);
    drawKeypoint.bindStream(&pipe[19], avp::AVP_STREAM_IN);
    drawKeypoint.bindStream(&pipe[14], avp::AVP_STREAM_IN);
    drawKeypoint.bindStream(&pipe[20], avp::AVP_STREAM_OUT);
    imshow_kp.bindStream(&pipe[20], avp::AVP_STREAM_IN);

    avp::StreamPacket inData(rawFrame, 0);
    pipe[0].loadPacket(inData);

    crop.process();
    std::cout<<"crop pass!\n";
    normalization.process();
    std::cout<<"normalization pass!\n";
    matToTensor.process();
    std::cout<<"matToTensor pass!\n";
    PalmCNN.process();
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
    rotateCropResize.process();
    std::cout<<"rotateCropResize pass!\n";
    normalization2.process();
    std::cout<<"normalization2 pass!\n";
    multiCropToTensor.process();
    std::cout<<"multiCropToTensor pass!\n";
    HandCNN.process();
    std::cout<<"HandCNN pass!\n";

    // int bs = pipe[11].front().size();
    // auto kps = pipe[17].front().tensor().reshape({bs, numKeypointsHand, 3});
    // auto kps_a = kps.accessor<float, 3>();
    // for(int i=0; i<bs; i++)
    // {
    //     for(int j=0; j<numKeypointsHand; j++)
    //     {
    //         cv::circle(pipe[11].front().mat(i), cv::Point(kps_a[i][j][0], kps_a[i][j][1]), 2, {0,0,255});
    //     }
    //     cv::imshow("", pipe[11].front().mat(i));
    //     cv::waitKey();
    // }

    rotateBack.process();
    std::cout<<"rotateBack pass!\n";
    // std::cout<<pipe[19].front().tensor().sizes()<<"\n";
    drawKeypoint.process();
    std::cout<<"drawKeypoint pass!\n";
    imshow_kp.process();
    std::cout<<"imshow_kp pass!\n";
}