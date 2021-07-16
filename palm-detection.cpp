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
int numKeypointsPalm = 7, numKeypointsHand = 21, numJointConnect=20, palmDetFreq = 20;
float scoreClipThrs = 100.0, minScoreThrs = 0.80, minSuppressionThrs=0.3, handThrs = 0.8;
float palm_shift_y = 0.5, palm_shift_x = 0, palm_box_scale = 2.6,
      hand_shift_y = 0, hand_shift_x = 0,  hand_box_scale = 2.1;
bool videoMode = true, showPalm = false;
int jointConnect[20][2] = {
    {0,1}, {1,2}, {2,3}, {3,4}, {0,5}, {5,6}, {6,7}, {7,8}, {0,9}, {9,10}, {10,11}, 
    {11,12}, {0,13}, {13,14}, {14,15}, {15,16}, {0,17}, {17,18}, {18,19}, {19,20}
};
int nonFingerId[] = {
    0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18
};
int handUpId = 9, handDownId = 0, palmUpId = 2, palmDownId = 0;

struct detRect
{
    cv::Mat img;
    cv::Mat_<float> affineMat;
    cv::Point2f center;
    detRect(cv::Mat& src, cv::Mat_<float>& mat, cv::Point2f& point):
        img(src), affineMat(mat), center(point)
    {}
};

struct detMeta
{
    int fid; // Frame id, pay attention to MAX_INT;
    int detType; // 0 is palm det, 1 is landmark det
    int xmin, ymin, xmax, ymax;
    float shift_x, shift_y, box_scale;
    cv::Point2f handUp, handDown;
    detMeta(int x_min, int y_min, int x_max, int y_max, 
        cv::Point2f& Up, cv::Point2f& Down, int type=0, int id=0):
        fid(id), xmin(x_min), ymin(y_min), xmax(x_max), ymax(y_max), detType(type), handUp(Up), handDown(Down)
    {
        if(type==0)
        {
            shift_x = palm_shift_x;
            shift_y = palm_shift_y;
            box_scale = palm_box_scale;
        }
        else
        {
            shift_x = hand_shift_x;
            shift_y = hand_shift_y;
            box_scale = hand_box_scale;
        }
    }
    detRect getTransformedRect(cv::Mat &img, bool square_long=true);
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
    deque<detRect> cropHands;
    deque<detMeta> handMetaForward;

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
    int showHeight = cropHeight, showWidth = cropWidth;
    int fid = 0;
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

        if(fid%palmDetFreq == 0)
        {
            handMetaForward.clear();
        }
        fid++;

        if(handMetaForward.empty())
        {
            cout<<"Palm Detection "<<fid<<endl;
            cv::resize(cropFrame, inFrame, cv::Size(modelWidth, modelHeight), 0, 0, cv::INTER_LINEAR);
            // showFrame = cropResize(frame, cropWidthLowBnd, cropHeightLowBnd, cropWidth, cropHeight);
            cv::cvtColor(inFrame, inFrame, cv::COLOR_BGR2RGB);
            inFrame.convertTo(inFrame, CV_32F);
            inFrame = inFrame / 127.5 - 1.0;
            // cout<<inFrame({0,0,4,4})<<endl;
            /* ---- NN Inference ---- */
            inputRawTensor = torch::from_blob(inFrame.data, {modelHeight, modelWidth, 3});
            inputTensor = torch::empty({batchSizePalm, 3, modelHeight, modelWidth}, torch::kF32);
            inputTensor[0] = inputRawTensor.permute({2,0,1});
            // cout<<inputTensor[0].slice(0, 0, 2)<<endl;
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
                        (float_t*)inputTensor.data_ptr(), palm_input_tensor_size, palm_input_node_dims.data(), 4);
            auto output_tensors = sessionPalm.Run(Ort::RunOptions{nullptr}, 
                        input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
            float* rawBoxesPPtr = output_tensors[0].GetTensorMutableData<float>();
            float* rawScoresPPtr = output_tensors[1].GetTensorMutableData<float>();
            // Decode Box
            rawBoxesP = torch::from_blob(rawBoxesPPtr, {batchSizePalm, numAnchors, outDim});
            rawScoresP = torch::from_blob(rawScoresPPtr, {batchSizePalm, numAnchors});

            // cout<<rawBoxesP.sum()<<endl;

            /* ---- Tensor to Detection ---- */
            decodeBoxes(rawBoxesP, anchors, detectionBoxes);
            auto detectionScores = rawScoresP.clamp(-scoreClipThrs, scoreClipThrs).sigmoid();
            auto mask = detectionScores >= minScoreThrs;
            // auto outBoxes = detectionBoxes.index(mask[0]);
            // cout<<mask.sizes()<<"\n"<<detectionBoxes.sizes()<<"\n"<<detectionScores.sizes()<<"\n\n";
            for(int i=0; i < batchSizePalm; i++)
            {
                auto boxes = detectionBoxes[i].index(mask[i]);
                auto scores = detectionScores[i].index(mask[i]).unsqueeze(-1);
                // rawDetections.push_back(torch::cat({boxes, scores}, -1));
                
                /* ---- NMS ---- */
                auto outDet = weightedNMS(torch::cat({boxes, scores}, -1));
                // outDetections.push_back(outDet);
                for(auto& det_t:outDet)
                {
                    auto det = det_t.accessor<float, 1>();
                    auto ymin = det[0] * showHeight;
                    auto xmin = det[1] * showWidth;
                    auto ymax = det[2] * showHeight;
                    auto xmax = det[3] * showWidth;
                    cv::Point2f handUp = cv::Point2f(det[4+palmUpId*2], det[4+palmUpId*2+1]),
                                handDown = cv::Point2f(det[4+palmDownId*2], det[4+palmDownId*2+1]);
                    handMetaForward.push_back(detMeta(xmin, ymin, xmax, ymax, handUp, handDown, 0));
                }
                // rawDetections.pop_front();
            }        
        }
        /* ---- Crop and Transform ---- */
        while(!handMetaForward.empty())
        {
            cropHands.push_back(handMetaForward.front().getTransformedRect(cropFrame));
            // cv::imshow("PalmDetection", cropHands.back().img);
            // cv::waitKey();
            handMetaForward.pop_front();
        }
        /* ---- Hand NN Inference ---- */
        showFrame = cropFrame;
        int batchSizeHand = cropHands.size();
        if(batchSizeHand)
        {
            std::vector<int64_t> hand_input_node_dims = {batchSizeHand, 3, modelHeight, modelWidth};
            size_t hand_input_tensor_size = batchSizeHand * 3 * modelHeight * modelWidth;
            int idx = 0;
            inputTensor = torch::empty({batchSizeHand, 3, modelWidth, modelHeight});
            for(auto& cropHand: cropHands)
            {
                cv::resize(cropHand.img, tmpFrame, cv::Size(modelWidth, modelHeight), 0, 0, cv::INTER_LINEAR);
                // showFrame = cropResize(frame, cropWidthLowBnd, cropHeightLowBnd, cropWidth, cropHeight);
                cv::cvtColor(tmpFrame, tmpFrame, cv::COLOR_BGR2RGB);
                tmpFrame.convertTo(tmpFrame, CV_32F);
                tmpFrame = tmpFrame / 127.5 - 1.0;
                auto tmpHand = torch::from_blob(tmpFrame.data, {modelHeight, modelWidth, 3});
                inputTensor[idx] = tmpHand.permute({2,0,1});
                idx++;
            }
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
                        (float_t*)inputTensor.data_ptr(), hand_input_tensor_size, hand_input_node_dims.data(), 4);
            auto output_tensors = sessionHand.Run(Ort::RunOptions{nullptr}, 
                        input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
            float* rawKeyptsHPtr = output_tensors[0].GetTensorMutableData<float>();
            float* rawScoresHPtr = output_tensors[1].GetTensorMutableData<float>();

            /* ---- Draw Hand landmarks ---- */
            size_t memOffset = numKeypointsHand * 3;
            for(int i=0; i<batchSizeHand; i++)
            {
                if(rawScoresHPtr[i]>handThrs)
                {
                    auto tmpWidth = cropHands.front().img.cols, tmpHeight = cropHands.front().img.rows;
                    cv::Mat_<float> keypointsHand=cv::Mat(numKeypointsHand, 3, CV_32F, (void*)(rawKeyptsHPtr+i*memOffset));
                    float x_offset = cropHands.front().center.x - tmpWidth * 0.5,
                          y_offset = cropHands.front().center.y - tmpHeight * 0.5;
                    keypointsHand = keypointsHand(cv::Rect(0,0,2,numKeypointsHand)).t();
                    keypointsHand.row(0) = keypointsHand.row(0) * tmpWidth / modelWidth + x_offset;
                    keypointsHand.row(1) = keypointsHand.row(1) * tmpHeight / modelHeight + y_offset;
                    auto keypointsMatRe = computePointAffine(keypointsHand, cropHands.front().affineMat, true);
                    float xmin, ymin, xmax, ymax;
                    xmin=xmax = keypointsMatRe(0,0);
                    ymin=ymax = keypointsMatRe(1,0);
                    int k = 0;
                    for(int j=0; j<numKeypointsHand; j++)
                    {
                        cv::circle(showFrame, cv::Point2f(keypointsMatRe(0,j), keypointsMatRe(1,j)), 4, {255, 0, 0}, -1);
                        // cv::circle(cropHands.front().img, cv::Point2f(keypointsHand(0,j)-x_offset+cropHands.front().affineMat.at<float>(0,2), 
                        //                             keypointsHand(1,j)-y_offset+cropHands.front().affineMat.at<float>(1,2)), 2, {0, 255, 0});
                        if(nonFingerId[k] == j)
                        {
                            xmin = min(xmin, keypointsMatRe(0,j));
                            xmax = max(xmax, keypointsMatRe(0,j));
                            ymin = min(ymin, keypointsMatRe(1,j));
                            ymax = max(ymax, keypointsMatRe(1,j));
                            k++;
                        }
                    }
                    // cv::imshow("PalmDetection", cropHands.front().img);
                    // cv::waitKey();
                    for(int j=0; j<numJointConnect; j++)
                    {
                        cv::line(showFrame, cv::Point2f(keypointsMatRe(0,jointConnect[j][0]), keypointsMatRe(1,jointConnect[j][0])), 
                                cv::Point2f(keypointsMatRe(0,jointConnect[j][1]), keypointsMatRe(1,jointConnect[j][1])), {255,255,255}, 2);
                    }
                    auto handUp = cv::Point2f(keypointsMatRe(0,handUpId), keypointsMatRe(1,handUpId));
                    auto handDown = cv::Point2f(keypointsMatRe(0,handDownId), keypointsMatRe(1,handDownId));
                    handMetaForward.push_back(detMeta(xmin, ymin, xmax, ymax, handUp, handDown, 1));
                }
                cropHands.pop_front();
            }
        }

        cv::imshow("PalmDetection", showFrame);
        if(!videoMode)
        {
            cv::waitKey();
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
    // cout<<detections.sizes()<<endl;
    vector<torch::Tensor> outDets;
    if(detections.size(0) == 0)
        return outDets;
    auto remaining = detections.slice(1,outDim, outDim+1).argsort(0, true).squeeze(-1);
    // cout<<remaining.sizes()<<"  "<<remaining;
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

detRect detMeta::getTransformedRect(cv::Mat &img, bool square_long)
{
    auto xscale = xmax - xmin, yscale = ymax - ymin;
    /* ---- Compute rotatation ---- */
    auto angleRad = atan2(handDown.x-handUp.x, handDown.y-handUp.y);
    auto angleDeg = angleRad*180/M_PI;
    // Movement
    // shift_y > 0 : move 0 --> 2; shift_x > 0 : move right hand side of 0->2
    auto x_center = xmin + xscale*(0.5-shift_y*sin(angleRad)+shift_x*cos(angleRad)); 
    auto y_center = ymin + yscale*(0.5-shift_y*cos(angleRad)-shift_x*sin(angleRad));

    if(square_long)
        xscale = yscale = max(xscale, yscale);

    auto xrescale = xscale*box_scale, yrescale = yscale*box_scale;
    /* ---- Get cropped Hands ---- */
    cv::Mat_<float> affineMat = cv::getRotationMatrix2D(cv::Point2f(img.cols, img.rows)/2, -angleDeg, 1);
    auto bbox = cv::RotatedRect(cv::Point2f(), img.size(), -angleDeg).boundingRect2f();
    affineMat.at<float>(0,2) += bbox.width/2.0 - img.cols/2.0;
    affineMat.at<float>(1,2) += bbox.height/2.0 - img.rows/2.0;
    cv::Mat rotFrame;
    cv::warpAffine(img, rotFrame, affineMat, bbox.size());                    
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
    return detRect(cropHand, affineMat, rotCenter);
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

        /* ---- Rotation & Crop ---- */
        // while(.empty())
        // {
        //     auto handLoc =.front();
        //     auto angleRad = atan2(handLoc.handDown.x-handLoc.handUp.x, handLoc.handDown.y-handLoc.handUp.y);
        //     auto angleDeg = angleRad*180/M_PI;
        //     float handScale = handLoc.getHandScale();
        //     float handRescale = handScale * box_scale;
        //     float handRescale_2 = handRescale / 2;
        //     auto palmCenter = (handLoc.handUp + handLoc.handDown) / 2;
        //     auto x_center = palmCenter.x - handScale*(palm_shift_y*sin(angleRad)+palm_shift_x*cos(angleRad)); 
        //     auto y_center = palmCenter.y - handScale*(palm_shift_y*cos(angleRad)-palm_shift_x*sin(angleRad));
        //     if(showPalm)
        //     {
        //         cv::Mat tmpImg;
        //         cropFrame.copyTo(tmpImg);
        //         // Line between point 0 and point 2
        //         cv::circle(tmpImg, palmCenter, 4, {0, 255, 255}, -1);
        //         cv::circle(tmpImg, cv::Point2f(x_center, y_center), 4, {0, 255, 0}, -1);
        //         cv::line(tmpImg, handLoc.handUp, handLoc.handDown, {255,255,255}, 1);
        //         auto rRect = cv::RotatedRect(cv::Point2f(x_center, y_center), cv::Size2f(handRescale, handRescale), 90 - angleDeg);
        //         cv::Point2f vertices[4];
        //         rRect.points(vertices);
        //         for (int i = 0; i < 4; i++)
        //             cv::line(tmpImg, vertices[i], vertices[(i+1)%4], {0,255,0}, 1);
        //         cv::imshow("PalmDetection", tmpImg);
        //         if(!videoMode)
        //         {
        //             cv::waitKey();
        //         }
        //     }
        //     /* ---- Get cropped Hands ---- */
        //     cv::Mat_<float> affineMat = cv::getRotationMatrix2D(cv::Point2f(showWidth, showHeight)/2, -angleDeg, 1);
        //     auto bbox = cv::RotatedRect(cv::Point2f(), cropFrame.size(), -angleDeg).boundingRect2f();
        //     affineMat.at<float>(0,2) += bbox.width/2.0 - cropFrame.cols/2.0;
        //     affineMat.at<float>(1,2) += bbox.height/2.0 - cropFrame.rows/2.0;
        //     cv::Mat rotFrame;
        //     cv::warpAffine(cropFrame, rotFrame, affineMat, bbox.size());   
        //     // Cropping & Point Affine Transformation
        //     cv::Mat_<float> pointMat(2,1);
        //     pointMat << x_center, y_center;
        //     // cout<<pointMat<<endl;
        //     cv::Mat_<float> rotPtMat = computePointAffine(pointMat, affineMat, false);
        //     // cout<<computePointAffine(rotPtMat, affineMat, true)<<endl;
        //     cv::Point2f rotCenter(rotPtMat(0), rotPtMat(1));
        //     // Out of range cases
        //     float xDwHalf = min(rotCenter.x, handRescale_2), yDwHalf = min(rotCenter.y, handRescale_2);
        //     float xUpHalf = rotCenter.x+handRescale_2 > rotFrame.cols?rotFrame.cols-rotCenter.x:handRescale_2;
        //     float yUpHalf = rotCenter.y+handRescale_2 > rotFrame.rows?rotFrame.rows-rotCenter.y:handRescale_2;
        //     auto cropHand = rotFrame(cv::Rect(rotCenter.x-xDwHalf, rotCenter.y-yDwHalf, xDwHalf+xUpHalf, yDwHalf+yUpHalf));
        //     cv::copyMakeBorder(cropHand, cropHand, handRescale_2-yDwHalf, handRescale_2-yUpHalf, 
        //                         handRescale_2-xDwHalf, handRescale_2-xUpHalf, cv::BORDER_CONSTANT);

        //     cropHands.push_back(cropHand);
        //     affineMats.push_back(affineMat);
        //     handCenters.push_back(rotCenter);
        //    .pop_front();
        // }
