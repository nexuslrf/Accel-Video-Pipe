#pragma once

#include "../avpipe/base.hpp"

namespace avp {

// For HRNet Pose estimation
class LandMarkMaxPred: public PipeProcessor {
public:
    bool outputProb;
    LandMarkMaxPred(bool output_prob = true, std::string pp_name=""): 
        PipeProcessor(1, 2, AVP_TENSOR, pp_name, STREAM_PROC), outputProb(output_prob)
    {
        if(!outputProb)
        {
            numOutStreams = 1;
        }   
    }
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
        if(outputProb)
            out_data_list[1].tensor = maxvals.squeeze(-1);
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
                    d1_x = fmax(d1_x, -1);
                    d1_y = fmax(d1_y, -1);
                    keyPoints[i][j][0] = x + d1_x;
                    keyPoints[i][j][1] = y + d1_y;
                }
            }
        }
        out_data_list[0].tensor = keyPoints;
    }
};

/* Used by hand detection tasks, 
 * other use case: Unknown..
 * Anchor File Format (from numpy array): 
 *  anchors.astype('float32').tofile('anchors.bin')
 */
class DecodeDetBoxes: public PipeProcessor {
public:
    int numAnchors;
    int dstHeight, dstWidth;
    int numKeypoints;
    torch::Tensor anchors;
    DecodeDetBoxes(int num_anchors, string anchor_file, int dst_h, int dst_w, int num_keypts, string pp_name=""): 
        PipeProcessor(1, 1, AVP_TENSOR, pp_name, STREAM_PROC), numAnchors(num_anchors), 
        dstHeight(dst_h), dstWidth(dst_w), numKeypoints(num_keypts)
    {
        anchors = torch::from_file(anchor_file, NULL, numAnchors * 4 * sizeof(float));
    }
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        auto rawBoxesP = in_data_list[0].tensor;
        auto boxes = out_data_list[0].tensor;

        auto x_center = rawBoxesP.slice(2,0,1) / dstWidth  * anchors.slice(1,2,3) + anchors.slice(1,0,1);
        auto y_center = rawBoxesP.slice(2,1,2) / dstHeight * anchors.slice(1,3,4) + anchors.slice(1,1,2);
        
        auto w = rawBoxesP.slice(2,2,3) / dstWidth  * anchors.slice(1,2,3);
        auto h = rawBoxesP.slice(2,3,4) / dstHeight * anchors.slice(1,3,4);

        boxes.slice(2,0,1) = y_center - h / 2; // ymin
        boxes.slice(2,1,2) = x_center - w / 2; // xmin
        boxes.slice(2,2,3) = y_center + h / 2; // ymax
        boxes.slice(2,3,4) = x_center + w / 2; // xmax

        int offset = 4 + numKeypoints * 2;
        boxes.slice(2,4,offset,2) = rawBoxesP.slice(2,4,offset,2) / dstWidth  * anchors.slice(1,2,3) + anchors.slice(1,0,1);
        boxes.slice(2,5,offset,2) = rawBoxesP.slice(2,5,offset,2) / dstHeight * anchors.slice(1,3,4) + anchors.slice(1,1,2);
    }
};

}