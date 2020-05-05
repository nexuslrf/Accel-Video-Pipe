#include <iostream>
#include <avpipe/tensor_compute.hpp>
#include <avpipe/cv_compute.hpp>
#include <avpipe/base.hpp>
#include <tensor_compute/det_postprocess.hpp>

int main()
{
    std::string modelDir = "C:\\Users\\Ruofan\\Programming\\Accel-Video-Pipe\\models";
    std::string videoPath = modelDir + "\\kunkun_nmsl.mp4";
    std::string model_name = modelDir + "\\pose_resnet_34_256x192";
    std::string model_path = model_name + ".zip";
    std::string filter_path = modelDir + "\\gaussian_modulation.zip";

    // auto rawFrame = cv::imread(testImg);
    // int rawHeight = rawFrame.rows;
    // int rawWidth = rawFrame.cols;
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
    crop.bindStream({&pipe[0]}, {&pipe[1]});
    normalization.bindStream({&pipe[1]}, {&pipe[2]});
    matToTensor.bindStream({&pipe[2]}, {&pipe[3]});
    CNN.bindStream({&pipe[3]}, {&pipe[4]});
    filter.bindStream({&pipe[4]}, {&pipe[5]});
    maxPred.bindStream({&pipe[4]}, {&pipe[6], &pipe[9]});
    getKeypoint.bindStream({&pipe[5]}, {&pipe[6], &pipe[7]});
    draw.bindStream({&pipe[7], &pipe[1], &pipe[9]}, {&pipe[8]});
    imshow.bindStream(&pipe[8], avp::AVP_STREAM_IN);

    // avp::StreamPacket inData(rawFrame, 0);
    // pipe[0].loadPacket(inData);
    bool empty = false;
    while(!empty)
    {
        videoSrc.process();
        empty = pipe[0].empty();
        crop.process();
        normalization.process();
        matToTensor.process();
        CNN.process();
        filter.process();
        maxPred.process();
        getKeypoint.process();
        draw.process();
        imshow.process();
    }
}