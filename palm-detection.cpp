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
int numAnchors = 2944, outDim = 18, batchSize = 1;
int numKeypoints = 7;
float scoreClipThrs = 100.0, minScoreThrs = 0.75, minSuppressionThrs=0.3;

cv::Mat cropResize(const cv::Mat& frame, int xMin, int yMin, int xCrop, int yCrop);
inline void matToTensor(cv::Mat const& src, torch::Tensor& out);
void decodeBoxes(const torch::Tensor &rawBoxes, const torch::Tensor &anchors, torch::Tensor &boxes);
vector<torch::Tensor> weightedNMS(const torch::Tensor &detections);
torch::Tensor computeIoU(const torch::Tensor &boxA, const torch::Tensor &boxB);

int main()
{
    string rootDir = "/Users/liangruofan1/Program/CV_Models/palm_detector/";
    string anchorFile = rootDir + "anchors.bin";
    string palmModel = rootDir + "palm_detection.onnx";
    string testImg = rootDir + "pics/palm_test2.jpg";

    fstream fin(anchorFile, ios::in | ios::binary);
    torch::NoGradGuard no_grad; // Disable back-grad buffering
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto anchors = torch::empty({numAnchors, 4}, options);
    auto detectionBoxes = torch::empty({batchSize, numAnchors, outDim});
    // load anchor binary file
    fin.read((char *)anchors.data_ptr(), anchors.numel() * sizeof(float));
    cout<<anchors.slice(0, 0, 6)<<endl;
    auto tmp = torch::from_file(anchorFile, NULL, anchors.numel(), options).reshape({numAnchors, 4});
    cout<<tmp.sizes()<<endl<<tmp.slice(0,0,6);
    // load ONNX model
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, palmModel.c_str(), session_options);
    auto frame = cv::imread(testImg);
    int rawHeight = frame.rows, rawWidth = frame.cols;
    int cropWidthLowBnd, cropWidth, cropHeightLowBnd, cropHeight;

    torch::Tensor inputRawTensor, inputTensor, rawBoxes, rawScores;
    deque<torch::Tensor> rawDetections;
    deque<vector<torch::Tensor>> outDetections;

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t> input_node_dims = {batchSize, 3, modelHeight, modelWidth};
    size_t input_tensor_size = batchSize * 3 * modelHeight * modelWidth;
    std::vector<float> input_tensor_values(input_tensor_size);
    std::vector<const char*> input_node_names = {"input"};
    std::vector<const char*> output_node_names = {"output1", "output2"};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // show pic
    // cv::imshow("PalmDetection", frame);
    // cv::waitKey();
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
    // cout<<cropFrame({0,0,4,4})<<endl;
    /* ---- NN Inference ---- */
    inputRawTensor = torch::from_blob(cropFrame.data, {modelHeight, modelWidth, 3});
    inputTensor = inputRawTensor.permute({2,0,1}).to(torch::kCPU, false, true);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
                (float_t*)inputTensor.data_ptr(), input_tensor_size, input_node_dims.data(), 4);
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
    cout<<output_tensors.size()<<"\n";
    float* rawBoxesPtr = output_tensors[0].GetTensorMutableData<float>();
    float* rawScoresPtr = output_tensors[1].GetTensorMutableData<float>();
    // for(int i=0; i<10; i++)
    //     cout<<rawBox[i]<<"  ";
    // Decode Box
    rawBoxes = torch::from_blob(rawBoxesPtr, {batchSize, numAnchors, outDim});
    rawScores = torch::from_blob(rawScoresPtr, {batchSize, numAnchors});

    /* ---- Tensor to Detection ---- */
    decodeBoxes(rawBoxes, anchors, detectionBoxes);
    // cout<<detectionBoxes.slice(1,0,1);
    auto detectionScores = rawScores.clamp(-scoreClipThrs, scoreClipThrs).sigmoid();
    auto mask = detectionScores >= minScoreThrs;
    cout<<mask.sum();
    // auto outBoxes = detectionBoxes.index(mask[0]);
    cout<<mask[0].sizes();
    
    for(int i=0; i < batchSize; i++)
    {
        auto boxes = detectionBoxes[i].index(mask[i]);
        auto scores = detectionScores[i].index(mask[i]).unsqueeze(-1);
        cout<<boxes.sizes()<<"  "<<scores.sizes()<<endl;
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
    // @TODO
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

void decodeBoxes(const torch::Tensor &rawBoxes, const torch::Tensor &anchors, torch::Tensor &boxes)
{
    auto x_center = rawBoxes.slice(2,0,1) / modelWidth  * anchors.slice(1,2,3) + anchors.slice(1,0,1);
    auto y_center = rawBoxes.slice(2,1,2) / modelHeight * anchors.slice(1,3,4) + anchors.slice(1,1,2);
    
    auto w = rawBoxes.slice(2,2,3) / modelWidth  * anchors.slice(1,2,3);
    auto h = rawBoxes.slice(2,3,4) / modelHeight * anchors.slice(1,3,4);

    boxes.slice(2,0,1) = y_center - h / 2; // ymin
    boxes.slice(2,1,2) = x_center - w / 2; // xmin
    boxes.slice(2,2,3) = y_center + h / 2; // ymax
    boxes.slice(2,3,4) = x_center + w / 2; // xmax

    int offset = 4 + numKeypoints * 2;
    boxes.slice(2,4,offset,2) = rawBoxes.slice(2,4,offset,2) / modelWidth  * anchors.slice(1,2,3) + anchors.slice(1,0,1);
    boxes.slice(2,5,offset,2) = rawBoxes.slice(2,5,offset,2) / modelHeight * anchors.slice(1,3,4) + anchors.slice(1,1,2);
}

vector<torch::Tensor> weightedNMS(const torch::Tensor &detections)
{
    vector<torch::Tensor> outDets;
    if(detections.size(0) == 0)
        return outDets;
    auto remaining = detections.slice(1,outDim, outDim+1).argsort(0, true).squeeze(-1);
    cout<<remaining.sizes()<<"  "<<remaining[0];
    cout<<detections[remaining[0]].sizes()<<"\n";
    // torch::Tensor IoUs;
    while (remaining.size(0)>0)
    {
        auto weightedDet = detections[remaining[0]].to(torch::kCPU, false, true);
        auto firstBox = detections[remaining[0]].slice(0,0,4).unsqueeze(0);
        auto otherBoxes = detections.index(remaining).slice(1,0,4);
        cout<<firstBox.sizes()<<"    "<<otherBoxes.sizes();
        auto IoUs = computeIoU(firstBox, otherBoxes);
        cout<<IoUs.sizes();
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
    cout<<areaA.sizes()<<"  "<<areaB.sizes()<<endl;
    auto unions = areaA + areaB - interX;
    return (interX / unions).squeeze(0);
}