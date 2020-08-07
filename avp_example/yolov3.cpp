#include <iostream>
#include <avpipe/tensor_compute.hpp>
#include <avpipe/cv_compute.hpp>
#include <avpipe/util_compute.hpp>
#include <avpipe/base.hpp>
#include <tensor_compute/det_postprocess.hpp>

int main()
{
    std::string model_path = "C:\\Users\\shoot\\Programming\\CV_Experiments\\tensorflow-yolo-v3\\frozen_darknet_yolov3_model";
    std::string image_path = "C:\\Users\\shoot\\Programming\\CV_Experiments\\PyTorch-YOLOv3\\data\\samples\\dog.jpg";
    std::string class_file = "C:\\Users\\shoot\\Programming\\CV_Experiments\\tensorflow-yolo-v3\\coco.names";
    int dstHeight = 416, dstWidth = 416;
    float probThres = 0.3;

    // avp::ImgFileProcessor videoSrc(image_path, false, false);
    avp::WebCamProcessor videoSrc(0, true);
    avp::PaddedResize crop(videoSrc.rawHeight, videoSrc.rawWidth, dstHeight, dstWidth);
    avp::ColorSpaceConverter bgr2rgb;
    avp::DataLayoutConvertion matToTensor;
    avp::OpenVinoProcessor CNN({1,3,416,416}, avp::NCHW, model_path, 2);
    avp::YOLOParser yoloRegion;
    avp::NonMaxSuppressionV2 NMS(80);
    avp::DrawDetBoxes drawDet(1.0/crop.ratio, 1.0/crop.ratio, -1.0*crop.top/crop.ratio, -1.0*crop.left/crop.ratio, 
        0, {0,255,0}, {1,0,3,2}, 5, 4, class_file);
    avp::StreamShowProcessor imshow(1);

    avp::Stream pipe[9];
    
    videoSrc.bindStream({}, {&pipe[0]});
    crop.bindStream({&pipe[0]}, {&pipe[1]});
    bgr2rgb.bindStream({&pipe[1]}, {&pipe[2]});
    matToTensor.bindStream({&pipe[2]}, {&pipe[3]});
    CNN.bindStream({&pipe[3]}, {&pipe[4], &pipe[5]});
    yoloRegion.bindStream({&pipe[4], &pipe[5]}, {&pipe[6]});
    NMS.bindStream({&pipe[6]}, {&pipe[7]});
    drawDet.bindStream({&pipe[7], &pipe[0]}, {&pipe[8]});
    imshow.bindStream({&pipe[8]}, {});

    bool empty = false;
    while(!empty)
    {
        videoSrc.process();
        crop.process();
        bgr2rgb.process();
        matToTensor.process();
        CNN.process();
        yoloRegion.process();
        NMS.process();
        drawDet.process();
        imshow.process();
    }
}