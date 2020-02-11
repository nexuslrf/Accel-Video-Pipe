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
// #ifdef CV_CXX11
#include <mutex>
#include <thread>
#include <queue>
// #endif

using namespace std;

int modelWidth=192, modelHeight=256;
int outWidth = 48, outHeight = 64;
int numJoints = 17, batchSize = 1;
int sigma = 2;
float eps = 1e-8;
cv::Scalar mean(0.485, 0.456, 0.406), stdev(0.229, 0.224, 0.225);
vector<int> flipOrder{0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15};

class KeyPoints {
    public:
        vector<cv::Point2i> coords;
        KeyPoints (int);
        void multiply(float);
};

cv::Mat cropResize(const cv::Mat& frame, int xMin, int yMin, int xCrop, int yCrop);
inline void updInput(vector<torch::jit::IValue>& inputs, torch::Tensor& inTensor);
inline void matToTensor(cv::Mat const& src, torch::Tensor& out);
void flipBack(torch::Tensor& src);
void getMaxPreds(torch::Tensor& heatmaps, torch::Tensor &preds, torch::Tensor& probs);
void getFinalPreds(torch::Tensor& heatmaps, torch::Tensor& coords, vector<KeyPoints>& keyPointsList);

int main()
{
    /* Variable declaration & initialization */
    // std variables
    int rawWidth, rawHeight, cropWidthLowBnd, cropWidth, cropHeightLowBnd, cropHeight;
    float fps=1, ratio, scale=4;

    // opencv variables
    cv::Mat frame, cropFrame, flipFrame, mergeFrame, showFrame, tmpFrame;
    cv::VideoCapture cap;
    vector<KeyPoints> keyPointsList(batchSize, KeyPoints(numJoints));
    static const std::string kWinName = "Deep learning in OpenCV";
    cv::namedWindow(kWinName, cv::WINDOW_NORMAL);

    // libTorch variables
    torch::NoGradGuard no_grad; // Disable back-grad buffering
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensorFrame_1, tensorFrame_2, modelInput, preds, probs,
                rawOut, output, outputDM;
    tensorFrame_1 = torch::empty({modelHeight, modelWidth, 3}, options);
    tensorFrame_2 = torch::empty({modelHeight, modelWidth, 3}, options);
    preds = torch::empty({batchSize, numJoints, 2}, torch::TensorOptions().dtype(torch::kI32));
    probs = torch::empty({batchSize, numJoints, 1}, options);
    vector<torch::jit::IValue> inputs;
    inputs.push_back(modelInput);
    // libTorch models
    torch::jit::script::Module model, gaussianModulation;
    model = torch::jit::load("../../HRNet-Human-Pose-Estimation/torchScript_pose_resnet_34_256x192.zip");
    gaussianModulation = torch::jit::load("../../HRNet-Human-Pose-Estimation/torchScript_gaussian_modulation.zip");

    /* Read source data */
//     
// /*    if (parser.has("input"))
//         cap.open(parser.get<String>("input"));
//     else
//         cap.open(parser.get<int>("device"));
// */ 
    cap.open("../kunkun_nmsl.mp4");
//     // cap.open(0);
//     // cap.set(CAP_PROP_FPS, 20);
    fps = cap.get(cv::CAP_PROP_FPS);
    rawWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    rawHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

#ifdef _IMG
    frame = cv::imread("/Users/liangruofan1/Program/CNN_Deployment/messi.jpg", 1);
    rawWidth = frame.cols, rawHeight = frame.rows;
#endif
    ratio = 1.0 * modelWidth / modelHeight;
    if (rawWidth * modelHeight > rawHeight * modelWidth)
    {
        float max_expand = rawHeight * ratio;
        cropHeightLowBnd = 0; 
        cropHeight = rawHeight;
        cropWidthLowBnd = (rawWidth - max_expand)/2;
        cropWidth = max_expand;
    }
    else
    {
        float max_expand = rawWidth / ratio;
        cropHeightLowBnd = (rawHeight - max_expand) / 2;
        cropHeight = max_expand;
        cropWidthLowBnd = 0;
        cropWidth = rawWidth;
    }
#ifdef _DEBUG
    cout<<"DataInfo:\nrawFPS: "<<fps<<"\nWidth: "<<rawWidth<<"\nHeight: "<<rawHeight<<endl;
    cout<<cropWidthLowBnd<<","<<cropWidth<<","<<cropHeightLowBnd<<","<<cropHeight<<endl;
#endif
    while (cv::waitKey(1) < 0)
    {
#ifdef _TIMING
        auto start = chrono::high_resolution_clock::now();
#endif
        cap >> frame;
        if (frame.empty())
        {
            cv::waitKey();
            break;
        } 
    // }
    
    /* OpenCV pre-processing */
    showFrame = cropResize(frame, cropWidthLowBnd, cropHeightLowBnd, cropWidth, cropHeight);
    cv::cvtColor(showFrame, tmpFrame, cv::COLOR_BGR2RGB);
    tmpFrame.convertTo(cropFrame, CV_32F);
    cropFrame = (cropFrame / 255 - mean) / stdev;
    cv::flip(cropFrame, flipFrame, +1);
#ifdef _TIMING
    auto stop1 = chrono::high_resolution_clock::now(); 
#endif
    /* LibTorch DNN model processing */
    matToTensor(cropFrame, tensorFrame_1);
    matToTensor(flipFrame, tensorFrame_2);
    modelInput = torch::stack({tensorFrame_1, tensorFrame_2}, 0);
    modelInput = modelInput.permute({0,3,1,2});
    updInput(inputs, modelInput);
    rawOut = model.forward(inputs).toTensor();
#ifdef _TIMING
    auto stop2 = chrono::high_resolution_clock::now(); 
#endif

#ifdef _DEBUG
    cout<<modelInput.sizes()<<" "<<modelInput[0][0][0][0]<<" "<<modelInput[1][0][0][0]<<endl;
    cout<<rawOut.sizes()<<endl;
#endif
    /* LibTorch & OpenCV Post processing */
    auto tmpOuts = rawOut.split(1, 0);
    flipBack(tmpOuts[1]);
    output = (tmpOuts[0] + tmpOuts[1]) * 0.5;
    updInput(inputs, output);
    outputDM = gaussianModulation.forward(inputs).toTensor();
    getMaxPreds(output, preds, probs);
    getFinalPreds(outputDM, preds, keyPointsList);
    for(auto& keyPoints: keyPointsList)
        keyPoints.multiply(scale);
#ifdef _TIMING
    auto stop3 = chrono::high_resolution_clock::now(); 
#endif
#ifdef _DEBUG
    auto unFB = torch::Tensor(tmpOuts[1]);
    cout<<outputDM.sizes()<<endl;
#endif
    /* OpenCV visualization */
    auto vis = probs > 0.3;
    auto vis_a = vis.accessor<bool, 3>();
    for(int i=0; i<numJoints; i++)
        if(vis_a[0][i][0])
            cv::circle(showFrame, keyPointsList[0].coords[i], 2, {0, 0, 255});
    cv::imshow(kWinName, showFrame);
#ifdef _TIMING
    auto stop4 = chrono::high_resolution_clock::now(); 
    cout<<"Timing Results:"
        <<"\nPreProcessing:  "<<chrono::duration_cast<chrono::milliseconds>(stop1-start).count()
        <<"\nNN Processing:  "<<chrono::duration_cast<chrono::milliseconds>(stop2-stop1).count()
        <<"\nPostProcessing: "<<chrono::duration_cast<chrono::milliseconds>(stop3-stop2).count()
        <<"\nVisualizing:    "<<chrono::duration_cast<chrono::milliseconds>(stop4-stop3).count()
        <<"\n";
#endif
    // cv::waitKey(0);
    }

    /////////////////For Test & Debug//////////////////////
#ifdef _FOO_TEST
    //// # flip test
    // int idx1=0, idx2, idx3, idx4;
    // while(idx1>=0)
    // {
    //     cin>>idx1>>idx2>>idx3>>idx4;
    //     cout<<modelInput[0][idx2][idx3][idx4]<<endl;
    //     cout<<modelInput[1][idx2][idx3][191-idx4]<<endl;
    //     cout<<tensorFrame_1[idx3][idx4][idx2]<<endl;
    //     cout<<tensorFrame_2[idx3][191-idx4][idx2]<<endl;
    //     cout<<unFB[0][idx2][idx3][idx4]<<endl;
    //     cout<<tmpOuts[1][0][idx2][idx3][idx4]<<endl;
    // }
    //// # slicing test
    // auto tA = torch::arange(24).reshape({4, 3, 2});
    // auto tB = torch::arange(24, 48).reshape({4, 3, 2});
    // tA.slice(1, 0, 1) = tB.slice(1, 0, 1);
    // cout<<tA<<endl<<(tA < 12);
    //// # getMaxPreds test
    // auto tC = torch::rand({1, numJoints, 64, 48}, options) - 0.5;
    // getMaxPreds(tC, preds, probs);
    // while(idx1>=0)
    // {
    //     cin>>idx1>>idx2>>idx3>>idx4;
    //     cout<<preds.slice(1, idx2, idx2+1)<<endl;
    //     cout<<probs.slice(1, idx2, idx2+1)<<endl;
    //     cout<<tC[idx1][idx2][idx3][idx4]<<endl;
    // }
    //// # accessor
    // auto tD = torch::arange(24, torch::TensorOptions().dtype(torch::kI32)).reshape({4, 3, 2});
    // auto tD_a = tD.accessor<int,3>();
    // cout<<tD_a[0][0][0]<<endl;
    // float x1 = 0.0012837 ,x2=0.00173969,x3=0.0022166 ,x4=0.00270054;
    // cout<<x1*x2*x3*x4<<endl;
    //// # Mat Scalar
    // vector<cv::Point2i> ps(5, cv::Point2i(4,8)); 
    // auto aM = cv::Mat(ps);
    // cv::Point2i z(3,4);
    // aM = aM * 2;
    // cout<<aM<<"\n"<<ps<<"\n\n"<<z*2;
#endif
}

KeyPoints::KeyPoints(int numPoints) {
    coords = vector<cv::Point2i>(numPoints);
}

void KeyPoints::multiply(float scale)
{
    for(auto& pt: coords)
        pt *= scale;
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

inline void updInput(vector<torch::jit::IValue>& inputs, torch::Tensor& inTensor)
{
    inputs.pop_back();
    inputs.push_back(inTensor);
}

inline void matToTensor(cv::Mat const& src, torch::Tensor& out)
{
    // stride=width*channels
    memcpy(out.data_ptr(), src.data, out.numel() * sizeof(float));
    return;
}

void flipBack(torch::Tensor& src)
{
    // NCHW
    src = src.flip(3);
    auto tmp = torch::empty_like(src);
    for(int i=0; i<(int)flipOrder.size(); i++)
        tmp.slice(1,i,i+1) = src.slice(1,flipOrder[i], flipOrder[i]+1);
    src.copy_(tmp);
}

void getMaxPreds(torch::Tensor& heatmaps, torch::Tensor &preds, torch::Tensor& probs)
{
    auto [maxvals, idx] =  heatmaps.reshape({heatmaps.size(0), heatmaps.size(1), -1}).max(2, true);
    // preds: [N, C, 2]
    preds.slice(2,0,1).copy_(idx % heatmaps.size(3));
    preds.slice(2,1,2).copy_(idx / heatmaps.size(3));
    auto predMusk = (maxvals > 0.0);
    preds.mul_(predMusk);
    probs.copy_(maxvals);
    // idx.slice({})    
}

void getFinalPreds(torch::Tensor& heatmaps, torch::Tensor& coords, vector<KeyPoints>& keyPointsList)
{
    auto map_a = heatmaps.accessor<float, 4>();
    auto xy_a = coords.accessor<int, 3>();
    int x, y;
    float d1_x, d1_y, d2;
    d2 = sigma * sigma / 4;
    for(int i=0; i<(int)heatmaps.size(0); i++)
    {
        for(int j=0; j<(int)heatmaps.size(1); j++)
        {
            x = xy_a[i][j][0];
            y = xy_a[i][j][1];
            if(x>0 && y>0)
            {
                d1_x = log(
                    (map_a[i][j][y][x+1] * map_a[i][j][y][x+1] * map_a[i][j][y+1][x+1] * map_a[i][j][y-1][x+1]) / 
                    (map_a[i][j][y][x-1] * map_a[i][j][y][x-1] * map_a[i][j][y+1][x-1] * map_a[i][j][y-1][x-1])
                ) * d2;
                d1_y = log(
                    (map_a[i][j][y+1][x] * map_a[i][j][y+1][x] * map_a[i][j][y+1][x+1] * map_a[i][j][y+1][x-1]) / 
                    (map_a[i][j][y-1][x] * map_a[i][j][y-1][x] * map_a[i][j][y-1][x+1] * map_a[i][j][y-1][x-1])
                ) * d2;
                keyPointsList[i].coords[j].x = x + d1_x;
                keyPointsList[i].coords[j].y = y + d1_y;
            }
        }
    }
    return;
}
