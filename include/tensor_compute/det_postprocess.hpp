#pragma once

#include "../avpipe/base.hpp"

namespace avp {

// For HRNet Pose estimation
class LandMarkMaxPred: public PipeProcessor {
public:
    LandMarkMaxPred(std::string pp_name=""): PipeProcessor(1, 1, AVP_TENSOR, pp_name, STREAM_PROC)
    {}
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        auto heatmaps = in_data_list[0].tensor;
        int bs = heatmaps.size(0), numJoints = heatmaps.size(1);
        auto [maxvals, idx] =  heatmaps.reshape({bs, numJoints, -1}).max(2, true);
        // preds: [N, C, 2]
        auto preds = torch::empty({bs, numJoints, 2}, torch::kI32);
        preds.slice(2,0,1) = idx % heatmaps.size(3);
        preds.slice(2,1,2) = idx / heatmaps.size(3);
        auto predMusk = (maxvals > 0.0);
        preds.mul_(predMusk);
        // probs.copy_(maxvals);
        out_data_list[0].tensor = preds;
    }
};

class PredToKeypoint: public PipeProcessor {
public:
    int sigma;
    PredToKeypoint(std::string pp_name=""): PipeProcessor(2, 1, AVP_TENSOR, pp_name, STREAM_PROC)
    {
        sigma = 2;
    }
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        auto heatmaps = in_data_list[0].tensor;
        int bs = heatmaps.size(0), numJoints = heatmaps.size(1);
        auto map_a = in_data_list[0].tensor.accessor<float, 4>();
        auto xy_a = in_data_list[1].tensor.accessor<int, 3>();
        auto keyPoints = torch::empty({bs, numJoints, 2}, torch::kI32);
        int x,y;
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
                    d1_x = std::log(
                        (map_a[i][j][y][x+1] * map_a[i][j][y][x+1] * map_a[i][j][y+1][x+1] * map_a[i][j][y-1][x+1]) / 
                        (map_a[i][j][y][x-1] * map_a[i][j][y][x-1] * map_a[i][j][y+1][x-1] * map_a[i][j][y-1][x-1])
                    ) * d2;
                    d1_y = std::log(
                        (map_a[i][j][y+1][x] * map_a[i][j][y+1][x] * map_a[i][j][y+1][x+1] * map_a[i][j][y+1][x-1]) / 
                        (map_a[i][j][y-1][x] * map_a[i][j][y-1][x] * map_a[i][j][y-1][x+1] * map_a[i][j][y-1][x-1])
                    ) * d2;
                    keyPoints[i][j][0] = x + d1_x;
                    keyPoints[i][j][1] = y + d1_y;
                }
            }
        }
        out_data_list[0].tensor = keyPoints;
    }
};

}