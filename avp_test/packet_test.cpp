#include <iostream>
#include <avpipe/base.hpp>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

using namespace std;

int main()
{
    avp::StreamPacket packet(avp::AVP_MAT);
    cv::Mat mat_data(4,5, CV_32FC3, {1,2,3});
    cout<<mat_data<<endl;
    packet.loadData(mat_data);
    cout<<packet.tensor().sizes()<<endl;
}