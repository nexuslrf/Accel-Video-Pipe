#pragma once

#include "../avpipe/base.hpp"

namespace avp {

class DrawLandMarks: public PipeProcessor{
public:
    float heightScale, widthScale;
    int heightOffset, widthOffset;
    float probThres;
    cv::Scalar color;
    DrawLandMarks(float h_scale=1.0, float w_scale=1.0, int h_offset=0, int w_offset=0, float prob_thres=0, cv::Scalar c={0,0,255}, std::string pp_name=""): 
        PipeProcessor(2, 1, AVP_MAT, pp_name, STREAM_PROC), heightScale(h_scale), widthScale(w_scale), 
        heightOffset(h_offset), widthOffset(w_offset), probThres(prob_thres), color(c)
    {
        if(probThres!=0)
            numInStreams = 3;
    }
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        auto rawCoords = in_data_list[0].tensor;
        auto probs = torch::ones({rawCoords.size(0), rawCoords.size(1)}, torch::kF32);
        if(probThres!=0)
            probs = in_data_list[2].tensor;
        auto probs_a = probs.accessor<float, 2>();
        cv::Mat showFrame;
        if(in_data_list[1].numConsume == 1)
            showFrame = in_data_list[1].mat;
        else
            in_data_list[1].mat.copyTo(showFrame);
            
        auto rawCoords_a = rawCoords.accessor<int, 3>();
        // idx 0 -> x idx 1 -> y
        int x, y;
        for(int i=0; i<rawCoords.size(0); i++)
        {
            for(int j=0; j<rawCoords.size(1); j++)
            {
                if(probs_a[i][j]>=probThres)
                {
                    x = rawCoords_a[i][j][0] * widthScale + widthOffset;
                    y = rawCoords_a[i][j][1] * heightScale + heightOffset;
                    cv::circle(showFrame, cv::Point(x,y), 2, color);
                }
            }
        }
        out_data_list[0].mat = showFrame;
    }
};

// /* 
//  *
//  */
// class DrawDetBoxes: public PipeProcessor {
// public:
//     int dstHeight, dstWidth;

//     DrawDetBoxes();
// };

}