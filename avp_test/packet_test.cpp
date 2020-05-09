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

    cout<<"---------------\n";
    avp::StreamPacket nullData(avp::AVP_TENSOR, 0);
    avp::Stream pipe[2];
    pipe[0].loadPacket(nullData);
    pipe[1].loadPacket(nullData);
    nullData.timestamp = 1;
    cout<<pipe[0].front().timestamp<<"\n";
    cout<<pipe[1].front().timestamp<<"\n";
}