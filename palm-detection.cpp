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
int numAnchors = 2944, outDim = 18, batchSizePalm = 1;
int numKeypointsPalm = 7, numKeypointsHand = 21, numJointConnect=20;
float scoreClipThrs = 100.0, minScoreThrs = 0.80, minSuppressionThrs=0.3, handThrs = 0.8;
float shift_y = 0.5, shift_x = 0, box_scale = 2.6;
bool videoMode = true, showPalm = false;
int jointConnect[20][2] = {
    {0,1}, {1,2}, {2,3}, {3,4}, {0,5}, {5,6}, {6,7}, {7,8}, {0,9}, {9,10}, {10,11}, 
    {11,12}, {0,13}, {13,14}, {14,15}, {15,16}, {0,17}, {17,18}, {18,19}, {19,20}
};

cv::Mat cropResize(const cv::Mat& frame, int xMin, int yMin, int xCrop, int yCrop);
inline void matToTensor(cv::Mat const& src, torch::Tensor& out);
void decodeBoxes(const torch::Tensor &rawBoxesP, const torch::Tensor &anchors, torch::Tensor &boxes);
vector<torch::Tensor> NMS(const torch::Tensor &detections);
vector<torch::Tensor> weightedNMS(const torch::Tensor &detections);
torch::Tensor computeIoU(const torch::Tensor &boxA, const torch::Tensor &boxB);
cv::Mat_<float> computePointAffine(cv::Mat_<float> &pointsMat, cv::Mat_<float> &affineMat, bool inverse);

int main()
{
    // file vars
    string rootDir = "/Users/liangruofan1/Program/CV_Models/";
    string anchorFile = rootDir + "palm_detector/anchors.bin";
    string palmModel = rootDir + "palm_detector/palm_detection.onnx";
    string handModel = rootDir + "hand_keypoint_3d/blaze_hand.onnx";
    string testImg = rootDir + "palm_detector/pics/LRF.jpg";

    // opencv vars
    int rawHeight, rawWidth, cropWidthLowBnd, cropWidth, cropHeightLowBnd, cropHeight;
    cv::Mat frame, rawFrame, showFrame, cropFrame, inFrame, tmpFrame;
    cv::VideoCapture cap;
    deque<cv::Mat> cropHands;
    deque<cv::Mat_<float>> affineMats;
    deque<cv::Point2f> handCenters;

    // libtorch vars
    torch::NoGradGuard no_grad; // Disable back-grad buffering
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto anchors = torch::empty({numAnchors, 4}, options);
    auto detectionBoxes = torch::empty({batchSizePalm, numAnchors, outDim});
    torch::Tensor inputRawTensor, inputTensor, rawBoxesP, rawScoresP, rawScoresH, rawKeyptsH;
    deque<torch::Tensor> rawDetections;
    deque<vector<torch::Tensor>> outDetections;

    /* ---- load anchor binary file ---- */
    fstream fin(anchorFile, ios::in | ios::binary);
    fin.read((char *)anchors.data_ptr(), anchors.numel() * sizeof(float));
    // cout<<anchors.slice(0, 0, 6)<<endl;
    // auto tmp = torch::from_file(anchorFile, NULL, anchors.numel(), options).reshape({numAnchors, 4});
    // cout<<tmp.sizes()<<endl<<tmp.slice(0,0,6);

    /* ---- init ONNX rt ---- */
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session sessionPalm(env, palmModel.c_str(), session_options);
    Ort::Session sessionHand(env, handModel.c_str(), session_options);
    
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t> palm_input_node_dims = {batchSizePalm, 3, modelHeight, modelWidth};
    size_t palm_input_tensor_size = batchSizePalm * 3 * modelHeight * modelWidth;
    // std::vector<float> input_tensor_values(palm_input_tensor_size);
    std::vector<const char*> input_node_names = {"input"};
    std::vector<const char*> output_node_names = {"output1", "output2"};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    /* ---- init opencv src file ---- */
    if(videoMode)
    {
        cap.open(0);
        rawWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        rawHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        // cap.set(cv::CAP_PROP_FPS, 20);
    }
    else
    {
        rawFrame = cv::imread(testImg);
        rawHeight = rawFrame.rows;
        rawWidth = rawFrame.cols;
        // show pic
        // cv::imshow("PalmDetection", rawFrame);
        // cv::waitKey();
    }

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
    // cout<<cropWidthLowBnd<<" "<<cropHeightLowBnd<<"\n";
    /* ---- OpenCV pre-processing ---- */
    while(cv::waitKey(1) < 0)
    {
        if(videoMode)
            cap >> rawFrame;
        if (rawFrame.empty())
        {
            cv::waitKey();
            break;
        } 
        // frame = rawFrame;
        cv::flip(rawFrame, frame, +1);
        cv::Rect ROI(cropWidthLowBnd, cropHeightLowBnd, cropWidth, cropHeight);
        cropFrame = frame(ROI);
        resize(frame(ROI), inFrame, cv::Size(modelWidth, modelHeight), 0, 0, cv::INTER_LINEAR);
        // showFrame = cropResize(frame, cropWidthLowBnd, cropHeightLowBnd, cropWidth, cropHeight);
        cv::cvtColor(inFrame, inFrame, cv::COLOR_BGR2RGB);
        inFrame.convertTo(inFrame, CV_32F);
        inFrame = inFrame / 127.5 - 1.0;
        // cout<<inFrame({0,0,4,4})<<endl;

        /* ---- NN Inference ---- */
        inputRawTensor = torch::from_blob(inFrame.data, {modelHeight, modelWidth, 3});
        inputTensor = inputRawTensor.permute({2,0,1}).to(torch::kCPU, false, true);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
                    (float_t*)inputTensor.data_ptr(), palm_input_tensor_size, palm_input_node_dims.data(), 4);
        auto output_tensors = sessionPalm.Run(Ort::RunOptions{nullptr}, 
                    input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
        // cout<<output_tensors.size()<<"\n";
        float* rawBoxesPPtr = output_tensors[0].GetTensorMutableData<float>();
        float* rawScoresPPtr = output_tensors[1].GetTensorMutableData<float>();
        // for(int i=0; i<10; i++)
        //     cout<<rawBox[i]<<"  ";
        // Decode Box
        rawBoxesP = torch::from_blob(rawBoxesPPtr, {batchSizePalm, numAnchors, outDim});
        rawScoresP = torch::from_blob(rawScoresPPtr, {batchSizePalm, numAnchors});

        /* ---- Tensor to Detection ---- */
        decodeBoxes(rawBoxesP, anchors, detectionBoxes);
        // cout<<detectionBoxes.slice(1,0,1);
        auto detectionScores = rawScoresP.clamp(-scoreClipThrs, scoreClipThrs).sigmoid();
        auto mask = detectionScores >= minScoreThrs;
        // cout<<mask.sum();
        // auto outBoxes = detectionBoxes.index(mask[0]);
        // cout<<mask[0].sizes();
        
        for(int i=0; i < batchSizePalm; i++)
        {
            auto boxes = detectionBoxes[i].index(mask[i]);
            auto scores = detectionScores[i].index(mask[i]).unsqueeze(-1);
            // cout<<boxes.sizes()<<"  "<<scores.sizes()<<endl;
            rawDetections.push_back(torch::cat({boxes, scores}, -1));

        }

        /* ---- NMS ---- */
        while(!rawDetections.empty())
        {
            auto outDet = weightedNMS(rawDetections.front());
            outDetections.push_back(outDet);
            rawDetections.pop_front();
        }

        /* ---- Show Res ---- */
        int showHeight = cropHeight, showWidth = cropWidth;
        while(!outDetections.empty())
        {
            // Re-scale boxes
            if(showPalm)
                cropFrame.copyTo(showFrame);
            else
                showFrame = cropFrame;
            
            auto outDets = outDetections.front();
            for(auto& det_t:outDets)
            {
                auto det = det_t.accessor<float, 1>();
                auto ymin = det[0] * showHeight;
                auto xmin = det[1] * showWidth;
                auto ymax = det[2] * showHeight;
                auto xmax = det[3] * showWidth;
                auto xscale = xmax - xmin, yscale = ymax - ymin;
                auto xrescale = xscale*box_scale, yrescale = yscale*box_scale;
                
                /* ---- Compute rotatation ---- */
                auto angleRad = atan2(det[4]-det[8], det[5]-det[9]);
                auto angleDeg = angleRad*180/M_PI;
                // Movement
                // shift_y > 0 : move 0 --> 2; shift_x > 0 : move right hand side of 0->2
                auto x_center = xmin + xscale*(0.5-shift_y*sin(angleRad)+shift_x*cos(angleRad)); 
                auto y_center = ymin + yscale*(0.5-shift_y*cos(angleRad)-shift_x*sin(angleRad));

                // Show Palm's Detection
                if(showPalm)
                {
                    cv::rectangle(showFrame, cv::Rect(xmin, ymin, xscale, yscale), {0, 0, 255});
                    for(int i=0; i < numKeypointsPalm; i++)
                    {
                        int offset = i * 2 + 4;
                        auto kp_x = det[offset  ] * showWidth;
                        auto kp_y = det[offset+1] * showHeight;
                        cv::circle(showFrame, cv::Point2f(kp_x, kp_y), 2, {0, 255, 0});
                    }
                    // Line between point 0 and point 2
                    cv::line(showFrame, cv::Point2f(det[4]*showWidth,det[5]*showHeight), 
                                    cv::Point2f(det[8]*showWidth,det[9]*showHeight), {255,255,255}, 1);
                    cv::rectangle(showFrame, cv::Rect(x_center-0.5*xscale, y_center-0.5*yscale, xscale, yscale), {0, 255, 255});
                    auto rRect = cv::RotatedRect(cv::Point2f(x_center, y_center), cv::Size2f(xrescale, yrescale), 90 - angleDeg);
                    cv::Point2f vertices[4];
                    rRect.points(vertices);
                    for (int i = 0; i < 4; i++)
                        cv::line(showFrame, vertices[i], vertices[(i+1)%4], {0,255,0}, 1);
                }
                /* ---- Get cropped Hands ---- */
                cv::Mat_<float> affineMat = cv::getRotationMatrix2D(cv::Point2f(showWidth, showHeight)/2, -angleDeg, 1);
                auto bbox = cv::RotatedRect(cv::Point2f(), cropFrame.size(), -angleDeg).boundingRect2f();
                affineMat.at<float>(0,2) += bbox.width/2.0 - cropFrame.cols/2.0;
                affineMat.at<float>(1,2) += bbox.height/2.0 - cropFrame.rows/2.0;
                cv::Mat rotFrame;
                cv::warpAffine(cropFrame, rotFrame, affineMat, bbox.size());                    
                // cv::imshow("Rotated", rotFrame);
                // cv::waitKey();
                // Cropping & Point Affine Transformation
                cv::Mat_<float> pointMat(2,1);
                pointMat << x_center, y_center;
                // cout<<pointMat<<endl;
                cv::Mat_<float> rotPtMat = computePointAffine(pointMat, affineMat, false);
                // cout<<computePointAffine(rotPtMat, affineMat, true)<<endl;
                cv::Point2f rotCenter(rotPtMat(0), rotPtMat(1));
                // Out of range cases
                float xrescale_2 = xrescale/2, yrescale_2 = yrescale/2;
                float xDwHalf = min(rotCenter.x, xrescale_2), yDwHalf = min(rotCenter.y, yrescale_2);
                float xUpHalf = rotCenter.x+xrescale_2 > rotFrame.cols?rotFrame.cols-rotCenter.x:xrescale_2;
                float yUpHalf = rotCenter.y+yrescale_2 > rotFrame.rows?rotFrame.rows-rotCenter.y:yrescale_2;
                auto cropHand = rotFrame(cv::Rect(rotCenter.x-xDwHalf, rotCenter.y-yDwHalf, xDwHalf+xUpHalf, yDwHalf+yUpHalf));
                cv::copyMakeBorder(cropHand, cropHand, yrescale_2-yDwHalf, yrescale_2-yUpHalf, 
                                    xrescale_2-xDwHalf, xrescale_2-xUpHalf, cv::BORDER_CONSTANT);
                cropHands.push_back(cropHand);
                affineMats.push_back(affineMat);
                handCenters.push_back(rotCenter);
            }
            int batchSizeHand = cropHands.size();
            if(batchSizeHand)
            {
                std::vector<int64_t> hand_input_node_dims = {batchSizeHand, 3, modelHeight, modelWidth};
                size_t hand_input_tensor_size = batchSizeHand * 3 * modelHeight * modelWidth;
                auto handsTensor = torch::empty({batchSizeHand, modelWidth, modelHeight, 3});
                int idx = 0;
                for(auto& cropHand: cropHands)
                {
                    resize(cropHand, tmpFrame, cv::Size(modelWidth, modelHeight), 0, 0, cv::INTER_LINEAR);
                    // showFrame = cropResize(frame, cropWidthLowBnd, cropHeightLowBnd, cropWidth, cropHeight);
                    cv::cvtColor(tmpFrame, tmpFrame, cv::COLOR_BGR2RGB);
                    tmpFrame.convertTo(tmpFrame, CV_32F);
                    tmpFrame = tmpFrame / 127.5 - 1.0;
                    auto tmpHand = torch::from_blob(tmpFrame.data, {1, modelHeight, modelWidth, 3});
                    handsTensor.slice(0, idx, idx+1) = tmpHand;
                    idx++;
                }
                /* ---- Hand NN Inference ---- */
                inputTensor = handsTensor.permute({0,3,1,2}).to(torch::kCPU, false, true);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
                            (float_t*)inputTensor.data_ptr(), hand_input_tensor_size, hand_input_node_dims.data(), 4);
                auto output_tensors = sessionHand.Run(Ort::RunOptions{nullptr}, 
                            input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
                // cout<<output_tensors.size()<<"\n";
                float* rawKeyptsHPtr = output_tensors[0].GetTensorMutableData<float>();
                float* rawScoresHPtr = output_tensors[1].GetTensorMutableData<float>();
                // rawKeyptsH = torch::from_blob(rawKeyptsHPtr, {batchSizeHand, numKeypointsHand, 3});
                // rawScoresH = torch::from_blob(rawScoresHPtr, {batchSizeHand});
                // cout<<rawScoresH<<endl;
                /* ---- Draw Hand landmarks ---- */
                // auto det = rawKeyptsH.accessor<float, 3>();
                size_t memOffset = numKeypointsHand * 3;
                for(int i=0; i<batchSizeHand; i++)
                {
                    if(rawScoresHPtr[i]>handThrs)
                    {
                        cv::Mat_<float> keypointsHand=cv::Mat(numKeypointsHand, 3, CV_32F, (void*)(rawKeyptsHPtr+i*memOffset));
                        float x_offset = handCenters.front().x - cropHands.front().cols * 0.5,
                            y_offset = handCenters.front().y - cropHands.front().rows * 0.5;
                        keypointsHand = keypointsHand(cv::Rect(0,0,2,numKeypointsHand)).t();
                        keypointsHand.row(0) = keypointsHand.row(0) * cropHands.front().cols / modelWidth + x_offset;
                        keypointsHand.row(1) = keypointsHand.row(1) * cropHands.front().rows / modelHeight + y_offset;
                        auto keypointsMatRe = computePointAffine(keypointsHand, affineMats.front(), true);
                        
                        for(int j=0; j<numKeypointsHand; j++)
                        {
                            cv::circle(showFrame, cv::Point2f(keypointsMatRe(0,j), keypointsMatRe(1,j)), 4, {255, 0, 0}, -1);
                            // cv::circle(cropHands.front(), cv::Point2f(keypointsHand(0,j)-x_offset+affineMats.front().at<float>(0,2), 
                            //                             keypointsHand(1,j)-y_offset+affineMats.front().at<float>(1,2)), 2, {0, 255, 0});
                        }
                        for(int j=0; j<numJointConnect; j++)
                        {
                            cv::line(showFrame, cv::Point2f(keypointsMatRe(0,jointConnect[j][0]), keypointsMatRe(1,jointConnect[j][0])), 
                                    cv::Point2f(keypointsMatRe(0,jointConnect[j][1]), keypointsMatRe(1,jointConnect[j][1])), {255,255,255}, 2);
                        }
                        // cv::imshow("CropHand", cropHands.front());
                        // cv::waitKey();
                    }
                    handCenters.pop_front(); affineMats.pop_front(); cropHands.pop_front();
                }
            }

            cv::imshow("PalmDetection", showFrame);
            if(!videoMode)
            {
                cv::waitKey();
            }

            outDetections.pop_front();
        }


    }
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

void decodeBoxes(const torch::Tensor &rawBoxesP, const torch::Tensor &anchors, torch::Tensor &boxes)
{
    auto x_center = rawBoxesP.slice(2,0,1) / modelWidth  * anchors.slice(1,2,3) + anchors.slice(1,0,1);
    auto y_center = rawBoxesP.slice(2,1,2) / modelHeight * anchors.slice(1,3,4) + anchors.slice(1,1,2);
    
    auto w = rawBoxesP.slice(2,2,3) / modelWidth  * anchors.slice(1,2,3);
    auto h = rawBoxesP.slice(2,3,4) / modelHeight * anchors.slice(1,3,4);

    boxes.slice(2,0,1) = y_center - h / 2; // ymin
    boxes.slice(2,1,2) = x_center - w / 2; // xmin
    boxes.slice(2,2,3) = y_center + h / 2; // ymax
    boxes.slice(2,3,4) = x_center + w / 2; // xmax

    int offset = 4 + numKeypointsPalm * 2;
    boxes.slice(2,4,offset,2) = rawBoxesP.slice(2,4,offset,2) / modelWidth  * anchors.slice(1,2,3) + anchors.slice(1,0,1);
    boxes.slice(2,5,offset,2) = rawBoxesP.slice(2,5,offset,2) / modelHeight * anchors.slice(1,3,4) + anchors.slice(1,1,2);
}

vector<torch::Tensor> weightedNMS(const torch::Tensor &detections)
{
    vector<torch::Tensor> outDets;
    if(detections.size(0) == 0)
        return outDets;
    auto remaining = detections.slice(1,outDim, outDim+1).argsort(0, true).squeeze(-1);
    // cout<<remaining.sizes()<<"  "<<remaining[0];
    // cout<<detections[remaining[0]].sizes()<<"\n";
    // torch::Tensor IoUs;
    while (remaining.size(0)>0)
    {
        auto weightedDet = detections[remaining[0]].to(torch::kCPU, false, true);
        auto firstBox = detections[remaining[0]].slice(0,0,4).unsqueeze(0);
        auto otherBoxes = detections.index(remaining).slice(1,0,4);
        // cout<<firstBox.sizes()<<"    "<<otherBoxes.sizes();
        auto IoUs = computeIoU(firstBox, otherBoxes);
        // cout<<IoUs.sizes();
        auto overlapping = remaining.index(IoUs > minSuppressionThrs);
        remaining = remaining.index(IoUs <= minSuppressionThrs);

        if(overlapping.size(0) > 1)
        {
            auto coords = detections.index(overlapping).slice(1,0,outDim);
            auto scores = detections.index(overlapping).slice(1,outDim, outDim+1);
            auto totalScore = scores.sum();
            weightedDet.slice(0,0,outDim) = (coords * scores).sum(0) / totalScore;
            weightedDet[outDim] = totalScore / overlapping.size(0);
        }
        outDets.push_back(weightedDet);
    }
    // cout<<outDets<<endl;
    return outDets;
}

vector<torch::Tensor> NMS(const torch::Tensor &detections)
{
    vector<torch::Tensor> outDets;
    if(detections.size(0) == 0)
        return outDets;
    auto remaining = detections.slice(1,outDim, outDim+1).argsort(0, true).squeeze(-1);
    // cout<<remaining.sizes()<<"  "<<remaining[0];
    // cout<<detections[remaining[0]].sizes()<<"\n";
    // torch::Tensor IoUs;
    while (remaining.size(0)>0)
    {
        auto Det = detections[remaining[0]].to(torch::kCPU, false, true);
        auto firstBox = detections[remaining[0]].slice(0,0,4).unsqueeze(0);
        auto otherBoxes = detections.index(remaining).slice(1,0,4);
        // cout<<firstBox.sizes()<<"    "<<otherBoxes.sizes();
        auto IoUs = computeIoU(firstBox, otherBoxes);
        // cout<<IoUs.sizes();
        auto overlapping = remaining.index(IoUs > minSuppressionThrs);
        remaining = remaining.index(IoUs <= minSuppressionThrs);

        outDets.push_back(Det);
    }
    // cout<<outDets<<endl;
    return outDets;
}

torch::Tensor computeIoU(const torch::Tensor &boxA, const torch::Tensor &boxB)
{
    // Compute Intersection
    int sizeA = boxA.size(0), sizeB = boxB.size(0);
    auto max_xy = torch::min(boxA.slice(1,2,4).unsqueeze(1).expand({sizeA, sizeB, 2}),
                             boxB.slice(1,2,4).unsqueeze(0).expand({sizeA, sizeB, 2}));
    auto min_xy = torch::max(boxA.slice(1,0,2).unsqueeze(1).expand({sizeA, sizeB, 2}),
                             boxB.slice(1,0,2).unsqueeze(0).expand({sizeA, sizeB, 2}));
    auto coords = (max_xy - min_xy).relu();
    auto interX = (coords.slice(2,0,1) * coords.slice(2,1,2)).squeeze(-1); // [sizeA, sizeB] 

    auto areaA = ((boxA.slice(1,2,3)-boxA.slice(1,0,1)) * 
                  (boxA.slice(1,3,4)-boxA.slice(1,1,2))).expand_as(interX);
    auto areaB = ((boxB.slice(1,2,3)-boxB.slice(1,0,1)) * 
                  (boxB.slice(1,3,4)-boxB.slice(1,1,2))).squeeze(-1).unsqueeze(0).expand_as(interX);
    // cout<<areaA.sizes()<<"  "<<areaB.sizes()<<endl;
    auto unions = areaA + areaB - interX;
    return (interX / unions).squeeze(0);
}

cv::Mat_<float> computePointAffine(cv::Mat_<float> &pointsMat, cv::Mat_<float> &affineMat, bool inverse)
{
    // cout<<pointsMat.size<<endl;
    if(!inverse)
    {
        cv::Mat_<float> ones = cv::Mat::ones(pointsMat.cols, 1, CV_32F);
        pointsMat.push_back(ones);
        return affineMat * pointsMat;
    }
    else
    {
        pointsMat.row(0)-=affineMat.at<float>(0,2);
        pointsMat.row(1)-=affineMat.at<float>(1,2);
        cv::Mat_<float> affineMatInv = affineMat(cv::Rect(0,0,2,2)).inv();
        return affineMatInv * pointsMat;
    }
}

        /* ---- Debug ---- */
        // cv::Mat debugFrame;
        // // Re-scale boxes
        // auto rawDets = rawDetections.front();
        // cout<<rawDets.sizes()<<endl;
        // auto det = rawDets.accessor<float, 2>();
        // for(int i=0; i<rawDets.size(0); i++)
        // {
        //     showFrame.copyTo(debugFrame);
        //     auto ymin = det[i][0] * modelHeight;
        //     auto xmin = det[i][1] * modelWidth;
        //     auto ymax = det[i][2] * modelHeight;
        //     auto xmax = det[i][3] * modelWidth;
        //     cv::rectangle(debugFrame, cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin), {0, 0, 255});

        //     for(int j=0; j < numKeypointsPalm; j++)
        //     {
        //         int offset = j * 2 + 4;
        //         auto kp_x = det[i][offset  ] * modelWidth;
        //         auto kp_y = det[i][offset+1] * modelHeight;
        //         cv::circle(debugFrame, cv::Point2f(kp_x, kp_y), 2, {0, 255, 0});
        //     }
        //     cv::imshow("PalmDetection", debugFrame);
        //     cv::waitKey();
        // }