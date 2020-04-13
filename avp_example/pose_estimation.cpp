#include <iostream>
#include <tensor_compute/libtorch.hpp>
#include <tensor_compute/openvino.hpp>
#include <tensor_compute/onnx_runtime.hpp>
#include <cv_compute/cv_stream_src.hpp>
#include <cv_compute/cv_display.hpp>
#include <cv_compute/cv_transfermation.hpp>
#include <tensor_compute/tensor_utils.hpp>
#include <tensor_compute/det_postprocess.hpp>
#include <cv_compute/cv_rendering.hpp>
#include <cv_compute/cv_stream_src.hpp>
#include <avpipe/base.hpp>

int main()
{
    std::string testImg = "/Users/liangruofan1/Program/Accel-Video-Pipe/test_data/cxk.png";
    std::string videoPath = "/Users/liangruofan1/Program/Accel-Video-Pipe/test_data/kunkun_nmsl.mp4";
    std::string model_name = "/Users/liangruofan1/Program/CV_Models/HRNet-Human-Pose-Estimation/pose_resnet_34_256x192";
    std::string model_path = model_name + ".zip";
    std::string filter_path = "/Users/liangruofan1/Program/CV_Models/HRNet-Human-Pose-Estimation/gaussian_modulation.zip";

    auto rawFrame = cv::imread(testImg);
    int rawHeight = rawFrame.rows;
    int rawWidth = rawFrame.cols;
    int dstHeight = 256, dstWidth = 192;
    float probThres = 0.3;

    avp::VideoFileProcessor videoSrc(videoPath);
    avp::CenterCropResize crop(videoSrc.rawHeight, videoSrc.rawWidth, dstHeight, dstWidth);
    avp::ImgNormalization normalization({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
    avp::DataLayoutConvertion matToTensor;
    avp::OpenVinoProcessor CNN({1,3,256,192}, avp::NCHW, model_name);
    avp::LibTorchProcessor filter({1,17,64,48}, avp::NCHW, filter_path);
    avp::LandMarkMaxPred maxPred;
    avp::PredToKeypoint getKeypoint;
    avp::DrawLandMarks draw(dstHeight/64, dstWidth/48, 0, 0, probThres);
    avp::StreamShowProcessor imshow(1);
    avp::Stream pipe[10];
    
    videoSrc.bindStream(&pipe[0], avp::AVP_STREAM_OUT);
    crop.bindStream(&pipe[0], avp::AVP_STREAM_IN);
    crop.bindStream(&pipe[1], avp::AVP_STREAM_OUT);
    normalization.bindStream(&pipe[1], avp::AVP_STREAM_IN);
    normalization.bindStream(&pipe[2], avp::AVP_STREAM_OUT);
    matToTensor.bindStream(&pipe[2], avp::AVP_STREAM_IN);
    matToTensor.bindStream(&pipe[3], avp::AVP_STREAM_OUT);
    CNN.bindStream(&pipe[3], avp::AVP_STREAM_IN);
    CNN.bindStream(&pipe[4], avp::AVP_STREAM_OUT);
    filter.bindStream(&pipe[4], avp::AVP_STREAM_IN);
    filter.bindStream(&pipe[5], avp::AVP_STREAM_OUT);
    maxPred.bindStream(&pipe[4], avp::AVP_STREAM_IN);
    maxPred.bindStream(&pipe[6], avp::AVP_STREAM_OUT);
    maxPred.bindStream(&pipe[9], avp::AVP_STREAM_OUT);
    getKeypoint.bindStream(&pipe[5], avp::AVP_STREAM_IN);
    getKeypoint.bindStream(&pipe[6], avp::AVP_STREAM_IN);
    getKeypoint.bindStream(&pipe[7], avp::AVP_STREAM_OUT);
    draw.bindStream(&pipe[7], avp::AVP_STREAM_IN);
    draw.bindStream(&pipe[1], avp::AVP_STREAM_IN);
    draw.bindStream(&pipe[9], avp::AVP_STREAM_IN);
    draw.bindStream(&pipe[8], avp::AVP_STREAM_OUT);
    imshow.bindStream(&pipe[8], avp::AVP_STREAM_IN);

    // avp::StreamPacket inData(rawFrame, 0);
    // pipe[0].loadPacket(inData);

    while(1)
    {
        videoSrc.process();
        // std::cout<<"video pass\n";
        crop.process();
        // std::cout<<"crop pass\n";
        normalization.process();
        // std::cout<<"normalization pass\n";
        matToTensor.process();
        // std::cout<<"matToTensor pass\n";
        CNN.process();
        // std::cout<<"CNN pass\n";
        filter.process();
        // std::cout<<"filter pass\n";
        maxPred.process();
        // std::cout<<"maxPred pass\n";
        getKeypoint.process();
        // std::cout<<"getKeypoint pass\n";
        draw.process();
        // std::cout<<"draw pass\n";
        imshow.process();
    }
}