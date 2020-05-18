#include <iostream>
#include <avpipe/tensor_compute.hpp>
#include <avpipe/cv_compute.hpp>
#include <avpipe/util_compute.hpp>
#include <avpipe/base.hpp>
#include <tensor_compute/det_postprocess.hpp>

int main()
{
    std::string rootDir = "C:\\Users\\Ruofan\\Programming\\Accel-Video-Pipe\\models\\";
    std::string anchorFile = rootDir + "anchors.bin";
    std::string palmModel = rootDir + "palm_detection.onnx";
    std::string handModel = rootDir + "blaze_hand.onnx";
    std::string testImg = rootDir + "LRF.jpg";

    auto rawFrame = cv::imread(testImg);
    int rawHeight = rawFrame.rows;
    int rawWidth = rawFrame.cols;
    int dstHeight = 256, dstWidth = 256;
    int numAnchors = 2944, numKeypointsPalm = 7, numKeypointsHand = 21;
    int handUpId = 9, handDownId = 0, palmUpId = 2, palmDownId = 0;
    float palm_shift_y = 0.5, palm_shift_x = 0, palm_box_scale = 2.6,
          hand_shift_y = 0, hand_shift_x = 0,  hand_box_scale = 2.1;

    avp::WebCamProcessor videoSrc;
    rawWidth = videoSrc.rawWidth; rawHeight = videoSrc.rawHeight;
    avp::CenterCropResize crop(rawHeight, rawWidth, dstHeight, dstWidth, false, true);
    avp::ImgNormalization normalization(0.5, 0.5);
    avp::DataLayoutConvertion matToTensor;
    avp::ONNXRuntimeProcessor PalmCNN({1,3,256,256}, avp::NCHW, palmModel, 2);
    avp::DecodeDetBoxes decodeBoxes(numAnchors, anchorFile, dstHeight, dstWidth, numKeypointsPalm);
    avp::NonMaxSuppression NMS(numKeypointsPalm);
    // avp::DrawDetBoxes drawDet(dstHeight, dstWidth);
    // avp::StreamShowProcessor imshow(-1, "Det");
    avp::RotateCropResize palmRotateCropResize(dstHeight, dstWidth, crop.cropHeight, crop.cropWidth,
                    palmUpId, palmDownId, palm_shift_y, palm_shift_x, palm_box_scale);
    avp::DataLayoutConvertion multiCropToTensor;
    avp::ImgNormalization normalization2(0.5, 0.5);
    avp::ONNXRuntimeProcessor HandCNN({0,3,256,256}, avp::NCHW, handModel, 2);
    avp::RotateBack rotateBack;
    avp::DrawLandMarks drawKeypoint;
    drawKeypoint.radius = 4;
    avp::StreamShowProcessor imshow_kp(100);
    avp::LandMarkToDet keypointToBndBox({0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18});
    avp::RotateCropResize handRotateCropResize(dstHeight, dstWidth, 1, 1, 
                    handUpId, handDownId, hand_shift_y, hand_shift_x, hand_box_scale);
    
    avp::TemplateProcessor multiplexer(2, 1, NULL, true, avp::AVP_MAT);
    multiplexer.bindFunc([&multiplexer](avp::DataList &in_data_list, avp::DataList &out_data_list) {
                   if (multiplexer.timeTick % 5 == 0 || in_data_list[0].empty())
                   {
                       std::cout << "----------palm branch!----------\n";
                       avp::Mat frame = in_data_list[1].mat();
                       out_data_list[0].loadData(frame);
                   }
               });

    avp::TemplateProcessor streamMerger(5, 4, NULL, true, avp::AVP_MAT);
    auto merger_func = [&](avp::DataList &in_data_list, avp::DataList &out_data_list) {
        if (!streamMerger.checkEmpty(0, 2))
        {
            auto palmStreams = avp::DataList({in_data_list[0], in_data_list[1], in_data_list[4]});
            palmRotateCropResize.run(palmStreams, out_data_list);
        }
        else if (!streamMerger.checkEmpty(2, 4))
        {
            std::cout << "hello wtf?\n";
            auto handStreams = avp::DataList({in_data_list[2], in_data_list[3], in_data_list[4]});
            handRotateCropResize.run(handStreams, out_data_list);
        }
    };
    streamMerger.bindFunc(merger_func);
    avp::TimeUpdater timeUpdate(2);

    avp::Stream pipe[30];
    videoSrc.bindStream({}, {&pipe[0]});
    crop.bindStream({&pipe[0]}, {&pipe[1], &pipe[14]});
    multiplexer.bindStream({&pipe[25], &pipe[1]}, {&pipe[23]});
    normalization.bindStream({&pipe[23]}, {&pipe[2]});
    matToTensor.bindStream({&pipe[2]}, {&pipe[3]});
    PalmCNN.bindStream({&pipe[3]}, {&pipe[4], &pipe[5]});
    decodeBoxes.bindStream({&pipe[4]}, {&pipe[6]});
    NMS.bindStream({&pipe[6], &pipe[5]}, {&pipe[7], &pipe[8]});
    // drawDet.bindStream({&pipe[7], &pipe[1]}, {&pipe[10]});
    // imshow.bindStream(&pipe[10], avp::AVP_STREAM_IN);
    streamMerger.bindStream({&pipe[7], &pipe[8], &pipe[25], &pipe[24], &pipe[14]}, 
                            {&pipe[11], &pipe[12], &pipe[13], &pipe[21]});
    normalization2.bindStream({&pipe[11]}, {&pipe[15]});
    multiCropToTensor.bindStream({&pipe[15]}, {&pipe[16]});
    HandCNN.bindStream({&pipe[16]}, {&pipe[17], &pipe[18]});
    rotateBack.bindStream({&pipe[17], &pipe[18], &pipe[21], &pipe[12], &pipe[13]}, {&pipe[19]});
    drawKeypoint.bindStream({&pipe[19], &pipe[14]}, {&pipe[20]});
    keypointToBndBox.bindStream({&pipe[19]}, {&pipe[22]});
    timeUpdate.bindStream({&pipe[19], &pipe[22]}, {&pipe[24], &pipe[25]});
    imshow_kp.bindStream({&pipe[20]}, {});

    avp::StreamPacket nullData(avp::AVP_TENSOR, 0);
    pipe[24].loadPacket(nullData);
    pipe[25].loadPacket(nullData); 
    // while(1)
    for(int i=0; i<1000; i++)
    {
        std::cout<<"----------Round: "<<i<<"----------\n";
        // auto frame = cv::imread(testImg);
        // avp::StreamPacket inData(frame, i);
        // pipe[0].loadPacket(inData);
        videoSrc.process();
        std::cout<<"videoSrc pass!\n";
        crop.process();
        std::cout<<"crop pass!\n";
        multiplexer.process();
        std::cout<<"multiplexer pass!\n";
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
        // drawDet.process();
        // std::cout<<"drawDet pass!\n";
        // imshow.process();
        // std::cout<<"imshow pass!\n";
        streamMerger.process();
        std::cout<<"streamMerger pass!\n";
        normalization2.process();
        std::cout<<"normalization2 pass!\n";
        multiCropToTensor.process();
        std::cout<<"multiCropToTensor pass!\n";
        HandCNN.process();
        std::cout<<"HandCNN pass!\n";
        rotateBack.process();
        std::cout<<"rotateBack pass!\n";
        drawKeypoint.process();
        std::cout<<"drawKeypoint pass!\n";
        keypointToBndBox.process();
        std::cout<<"keypointToBndBox pass!\n";
        timeUpdate.process();
        std::cout<<"timeUpdate pass!\n";
        // For debug
        // auto det_boxes = pipe[25].front().tensor();
        // std::cout<<det_boxes<<"\n";
        // int bs = det_boxes.size(0);
        // auto det_boxes_a = det_boxes.accessor<float, 2>();
        // for(int j=0; j<bs; j++)
        // {
        //     auto ymin = det_boxes_a[j][0];
        //     auto xmin = det_boxes_a[j][1];
        //     auto ymax = det_boxes_a[j][2];
        //     auto xmax = det_boxes_a[j][3];
        //     cv::rectangle(pipe[20].front().mat(), cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin), {0,255,0});
        // }
        // ---
        imshow_kp.process();
        std::cout<<"imshow_kp pass!\n";
    }
}