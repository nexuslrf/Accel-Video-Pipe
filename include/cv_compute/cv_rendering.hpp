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
    int probIdx, classIdx;
    float probThres;
    cv::Scalar color;
    std::vector<int> idxOrder;
    std::vector<std::string> classes;
    DrawDetBoxes(float h_scale=1.0, float w_scale=1.0, int h_offset=0, int w_offset=0, float prob_thres=0, 
        cv::Scalar c={0,255,0}, std::vector<int> idx_order={0,1,2,3}, int class_idx=-1, int prob_idx=-1, 
        std::string class_file="", std::string pp_name="DrawDetBoxes"): 
        PipeProcessor(2, 1, AVP_MAT, pp_name, STREAM_PROC), heightScale(h_scale), widthScale(w_scale), 
        heightOffset(h_offset), widthOffset(w_offset), probThres(prob_thres), color(c), idxOrder(idx_order),
        probIdx(prob_idx), classIdx(class_idx)
    {
        if(class_file!="")
        {
            std::fstream fin(class_file);
            std::string line;
            while(std::getline(fin, line))
            {
                classes.push_back(line);
            }
            fin.close();
        }
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
                auto ymin = rawBndBoxes_a[i][idxOrder[0]] * heightScale;
                auto xmin = rawBndBoxes_a[i][idxOrder[1]] * widthScale;
                auto ymax = rawBndBoxes_a[i][idxOrder[2]] * heightScale;
                auto xmax = rawBndBoxes_a[i][idxOrder[3]] * widthScale;
                cv::rectangle(showFrame, cv::Rect(xmin+widthOffset, ymin+heightOffset, xmax-xmin, ymax-ymin), color);
                std::string text = "";
                if(classIdx >= 0)
                {
                    int class_id = rawBndBoxes_a[i][classIdx];
                    if(classes.size())
                        text += (" class: "+classes[class_id]);
                    else
                        text += (" class: "+std::to_string(class_id));
                }
                if(probIdx >= 0)
                {
                    float prob = rawBndBoxes_a[i][probIdx];
                    text += (" prob: "+std::to_string(prob));
                }
                cv::putText(showFrame, text, cv::Point(xmin+widthOffset, ymin+heightOffset), 
                    cv::FONT_HERSHEY_COMPLEX, 0.6, color, 1);
                // std::cout<<text<<"\n";
            }
        }
        out_data_list[0].loadData(showFrame);
    }
};
}