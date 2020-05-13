#pragma once

#include "../avpipe/base.hpp"

namespace avp {

class DrawLandMarks: public PipeProcessor{
public:
    float heightScale, widthScale;
    int heightOffset, widthOffset;
    int radius;
    float probThres;
    Tensor probs;
    cv::Scalar color;
    DrawLandMarks(float h_scale=1.0, float w_scale=1.0, int h_offset=0, int w_offset=0, float prob_thres=0, 
        int r=2, cv::Scalar c={0,0,255}, std::string pp_name="DrawLandMarks"): 
        PipeProcessor(2, 1, AVP_MAT, pp_name, STREAM_PROC), heightScale(h_scale), widthScale(w_scale), 
        heightOffset(h_offset), widthOffset(w_offset), radius(r), probThres(prob_thres), color(c)
    {
        if(probThres!=0)
            numInStreams = 3;
        skipEmptyCheck = true;
    }
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        // must consider empty case when skipEmptyCheck is true;
        if(in_data_list[0].empty())
        {
            out_data_list[0].loadData(in_data_list[1].mat());
            return;
        }
        auto rawCoords = in_data_list[0].tensor();
        if(probThres!=0)
            probs = in_data_list[2].tensor();
        else
            probs = torch::ones({rawCoords.size(0), rawCoords.size(1)}, torch::kF32);
        auto probs_a = probs.accessor<float, 2>();
        cv::Mat showFrame;
        if(in_data_list[1].numConsume == 1)
            showFrame = in_data_list[1].mat();
        else
            in_data_list[1].mat().copyTo(showFrame);
        rawCoords = rawCoords.toType(torch::kF32);
        auto rawCoords_a = rawCoords.accessor<float, 3>();
        // idx 0 -> x idx 1 -> y
        int x, y;
        for(int i=0; i<rawCoords.size(0); i++)
        {
            for(int j=0; j<rawCoords.size(1); j++)
            {
                if(probs_a[i][j]>probThres)
                {
                    x = rawCoords_a[i][j][0] * widthScale + widthOffset;
                    y = rawCoords_a[i][j][1] * heightScale + heightOffset;
                    cv::circle(showFrame, cv::Point(x,y), radius, color);
                }
            }
        }
        out_data_list[0].loadData(showFrame);
    }
};

/* 
 *
 */
class DrawDetBoxes: public PipeProcessor {
public:
    float heightScale, widthScale;
    int heightOffset, widthOffset;
    float probThres;
    cv::Scalar color;
    DrawDetBoxes(float h_scale=1.0, float w_scale=1.0, int h_offset=0, int w_offset=0, 
        float prob_thres=0, cv::Scalar c={0,255,0}, std::string pp_name="DrawDetBoxes"): 
        PipeProcessor(2, 1, AVP_MAT, pp_name, STREAM_PROC), heightScale(h_scale), widthScale(w_scale), 
        heightOffset(h_offset), widthOffset(w_offset), probThres(prob_thres), color(c)
    {
        if(probThres!=0)
            numInStreams = 3;
        skipEmptyCheck = true;
    }
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        // must consider empty case when skipEmptyCheck is true;
        if(in_data_list[0].empty())
        {
            out_data_list[0].loadData(in_data_list[1].mat());
            return;
        }
        auto rawBndBoxes = in_data_list[0].tensor();
        auto probs = torch::ones({rawBndBoxes.size(0)}, torch::kF32);
        if(probThres!=0)
            probs = in_data_list[2].tensor();
        auto probs_a = probs.accessor<float, 1>();
        cv::Mat showFrame;
        if(in_data_list[1].numConsume == 1)
            showFrame = in_data_list[1].mat();
        else
            in_data_list[1].mat().copyTo(showFrame);
            
        auto rawBndBoxes_a = rawBndBoxes.accessor<float, 2>();
        for(int i=0; i<rawBndBoxes.size(0); i++)
        {
            if(probs_a[i]>probThres)
            {
                auto ymin = rawBndBoxes_a[i][0] * heightScale;
                auto xmin = rawBndBoxes_a[i][1] * widthScale;
                auto ymax = rawBndBoxes_a[i][2] * heightScale;
                auto xmax = rawBndBoxes_a[i][3] * widthScale;
                cv::rectangle(showFrame, cv::Rect(xmin+widthOffset, ymin+heightOffset, xmax-xmin, ymax-ymin), color);
            }
        }
        out_data_list[0].loadData(showFrame);
    }
};

}