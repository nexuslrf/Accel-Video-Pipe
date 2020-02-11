#include <fstream>
#include <sstream>
#include <cmath>
#include <functional> 
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

int main()
{
    // Open a video file or an image file or a camera stream.
//     VideoCapture cap;
// /*    if (parser.has("input"))
//         cap.open(parser.get<String>("input"));
//     else
//         cap.open(parser.get<int>("device"));
// */ 
//     cap.open("../kunkun_nmsl.mp4");
//     // cap.open(0);
//     // cap.set(CAP_PROP_FPS, 20);
//     double fps = cap.get(CAP_PROP_FPS);
//     int rawWidth = cap.get(CAP_PROP_FRAME_WIDTH);
//     int rawHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
    cv::Mat frame, cropFrame, flipFrame, mergeFrame, showFrame, tmpFrame;

    frame = cv::imread("/Users/liangruofan1/Program/CNN_Deployment/messi.jpg", 1);
    double fps = 1;
    int rawWidth = frame.cols, rawHeight = frame.rows;

    int cropWidthLowBnd, cropWidth, cropHeightLowBnd, cropHeight;
    float ratio = 1.0 * modelWidth / modelHeight;

    cout<<"rawFPS: "<<fps<<"\nWidth: "<<rawWidth<<"\nHeight: "<<rawHeight<<endl;
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
    cout<<cropWidthLowBnd<<","<<cropWidth<<","<<cropHeightLowBnd<<","<<cropHeight<<endl;
    
    static const std::string kWinName = "Deep learning in OpenCV";
    cv::namedWindow(kWinName, cv::WINDOW_NORMAL);

    // while (waitKey(1) < 0)
    // {
    //     cap >> frame;
    //     if (frame.empty())
    //     {
    //         waitKey();
    //         break;
    //     }
    //     imshow(kWinName, cropResize(frame, cropWidthLowBnd, cropHeightLowBnd, cropWidth, cropHeight));
         
    // }
    showFrame = cropResize(frame, cropWidthLowBnd, cropHeightLowBnd, cropWidth, cropHeight);
    // cv::imshow(kWinName, showFrame);
    cv::cvtColor(showFrame, tmpFrame, cv::COLOR_BGR2RGB);
    tmpFrame.convertTo(cropFrame, CV_32F);
    cropFrame = (cropFrame / 255 - mean) / stdev;
    cv::flip(cropFrame, flipFrame, +1);
    // cv::hconcat(cropFrame, flipFrame, mergeFrame);
    
    //////////LibTorch/////////
    torch::NoGradGuard no_grad; // Disable grad buffering
    // torch::Tensor modelInput = torch::from_blob(input.data, {1,input.rows, input.cols,3}, torch::kByte);
    // modelInput = modelInput.permute({0,3,1,2});
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor tensorFrame_1, tensorFrame_2, modelInput, preds, probs;
    tensorFrame_1 = torch::empty({modelHeight, modelWidth, 3}, options);
    tensorFrame_2 = torch::empty({modelHeight, modelWidth, 3}, options);
    preds = torch::empty({batchSize, numJoints, 2}, torch::TensorOptions().dtype(torch::kI32));
    probs = torch::empty({batchSize, numJoints, 1}, options);
    matToTensor(cropFrame, tensorFrame_1);
    matToTensor(flipFrame, tensorFrame_2);
    modelInput = torch::stack({tensorFrame_1, tensorFrame_2}, 0);
    modelInput = modelInput.permute({0,3,1,2});
    cout<<modelInput.sizes()<<" "<<modelInput[0][0][0][0]<<" "<<modelInput[1][0][0][0]<<endl;

    torch::jit::script::Module model, gaussianModulation;
    model = torch::jit::load("../../HRNet-Human-Pose-Estimation/torchScript_pose_resnet_34_256x192.zip");
    gaussianModulation = torch::jit::load("../../HRNet-Human-Pose-Estimation/torchScript_gaussian_modulation.zip");

    vector<torch::jit::IValue> inputs;
    inputs.push_back(modelInput);
    updInput(inputs, modelInput);
    auto rawOut = model.forward(inputs).toTensor();
    // cout<<rawOut.sizes()<<endl;
    auto tmpOuts = rawOut.split(1, 0);
    // auto unFB = torch::Tensor(tmpOuts[1]);
    flipBack(tmpOuts[1]);
    auto output = (tmpOuts[0] + tmpOuts[1]) * 0.5;
    updInput(inputs, output);
    auto outputDM = gaussianModulation.forward(inputs).toTensor();
    // cout<<outputDM.sizes()<<endl;
    getMaxPreds(output, preds, probs);
    // cv::waitKey(0);
    vector<KeyPoints> keyPointsList(batchSize, KeyPoints(numJoints));
    getFinalPreds(outputDM, preds, keyPointsList);

    float scale = 4; //cropHeight * 4 / modelHeight;
    for(auto& keyPoints: keyPointsList)
        keyPoints.multiply(scale);

    auto vis = probs > 0.3;
    auto vis_a = vis.accessor<bool, 3>();
    for(int i=0; i<numJoints; i++)
        if(vis_a[0][i][0])
            cv::circle(showFrame, keyPointsList[0].coords[i], 2, {0, 0, 255});
    
    cv::imshow(kWinName, showFrame);
    cv::waitKey(0);


    // cv::multiply
    /////////////////For Test & Debug//////////////////////
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
}