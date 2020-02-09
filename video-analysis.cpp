#include <fstream>
#include <sstream>
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

int modelWidth=192;
int modelHeight=256;
cv::Scalar mean(0.485, 0.456, 0.406), stdev(0.229, 0.224, 0.225);

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
    cv::imshow(kWinName, showFrame);
    cv::cvtColor(showFrame, tmpFrame, cv::COLOR_BGR2RGB);
    tmpFrame.convertTo(cropFrame, CV_32F);
    cropFrame = (cropFrame / 255 - mean) / stdev;
    cv::flip(cropFrame, flipFrame, +1);
    // cv::hconcat(cropFrame, flipFrame, mergeFrame);
    
    ///////////////////
    // torch::Tensor modelInput = torch::from_blob(input.data, {1,input.rows, input.cols,3}, torch::kByte);
    // modelInput = modelInput.permute({0,3,1,2});
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor tensorFrame_1, tensorFrame_2, modelInput, output;
    tensorFrame_1 = torch::empty({modelHeight, modelWidth, 3}, options);
    tensorFrame_2 = torch::empty({modelHeight, modelWidth, 3}, options);
    matToTensor(cropFrame, tensorFrame_1);
    matToTensor(flipFrame, tensorFrame_2);
    modelInput = torch::stack({tensorFrame_1, tensorFrame_2}, 0);
    modelInput = modelInput.permute({0,3,1,2});
    cout<<modelInput.sizes()<<" "<<modelInput[0][0][0][0]<<" "<<modelInput[1][0][0][0]<<endl;

    torch::jit::script::Module model;
    model = torch::jit::load("../../HRNet-Human-Pose-Estimation/torchScript_pose_resnet_34_256x192.zip");

    vector<torch::jit::IValue> inputs;
    inputs.push_back(modelInput);
    output = model.forward(inputs).toTensor();
    cout<<output.sizes()<<endl;

    // cv::waitKey(0);
    // int idx1=0, idx2, idx3, idx4;
    // while(idx1>=0)
    // {
    //     cin>>idx1>>idx2>>idx3>>idx4;
    //     cout<<modelInput[0][idx2][idx3][idx4]<<endl;
    //     cout<<modelInput[1][idx2][idx3][191-idx4]<<endl;
    //     cout<<tensorFrame_1[idx3][idx4][idx2]<<endl;
    //     cout<<tensorFrame_2[idx3][191-idx4][idx2]<<endl;
    // }
    
}