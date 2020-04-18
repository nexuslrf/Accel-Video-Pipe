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

torch::Tensor computeIoU(const torch::Tensor &boxA, const torch::Tensor &boxB)
{
    // Compute Intersection
    int sizeA = boxA.size(0), sizeB = boxB.size(0);
    auto max_xy = torch::min(boxA.slice(1,2,4).unsqueeze(1).expand({sizeA, sizeB, 2}),
                             boxB.slice(1,2,4).unsqueeze(0).expand({sizeA, sizeB, 2}));
    auto min_xy = torch::max(boxA.slice(1,0,2).unsqueeze(1).expand({sizeA, sizeB, 2}),
                             boxB.slice(1,0,2).unsqueeze(0).expand({sizeA, sizeB, 2}));
    auto coords = (max_xy - min_xy).relu();
    auto interX = (coords.slice(2,0,1) * coords.slice(2,1,2)).squeeze(-1); // [sizeA, sizeB] 

    auto areaA = ((boxA.slice(1,2,3)-boxA.slice(1,0,1)) * 
                  (boxA.slice(1,3,4)-boxA.slice(1,1,2))).expand_as(interX);
    auto areaB = ((boxB.slice(1,2,3)-boxB.slice(1,0,1)) * 
                  (boxB.slice(1,3,4)-boxB.slice(1,1,2))).squeeze(-1).unsqueeze(0).expand_as(interX);
    // cout<<areaA.sizes()<<"  "<<areaB.sizes()<<endl;
    auto unions = areaA + areaB - interX;
    return (interX / unions).squeeze(0);
}

void weightedNMS(const Tensor &detections, std::vector<Tensor>& outDets, int scoreDim, float minSuppressionThrs)
{
    if(detections.size(0) == 0)
        return;
    auto remaining = detections.slice(1,scoreDim, scoreDim+1).argsort(0, true).squeeze(-1);
    // cout<<remaining.sizes()<<"  "<<remaining[0];
    // cout<<detections[remaining[0]].sizes()<<"\n";
    // torch::Tensor IoUs;
    while (remaining.size(0)>0)
    {
        auto weightedDet = detections[remaining[0]].to(torch::kCPU, false, true);
        auto firstBox = detections[remaining[0]].slice(0,0,4).unsqueeze(0);
        auto otherBoxes = detections.index(remaining).slice(1,0,4);
        // cout<<firstBox.sizes()<<"    "<<otherBoxes.sizes();
        auto IoUs = computeIoU(firstBox, otherBoxes);
        // cout<<IoUs.sizes();
        auto overlapping = remaining.index(IoUs > minSuppressionThrs);
        remaining = remaining.index(IoUs <= minSuppressionThrs);

        if(overlapping.size(0) > 1)
        {
            auto coords = detections.index(overlapping).slice(1,0,scoreDim);
            auto scores = detections.index(overlapping).slice(1,scoreDim, scoreDim+1);
            auto totalScore = scores.sum();
            weightedDet.slice(0,0,scoreDim) = (coords * scores).sum(0) / totalScore;
            weightedDet[scoreDim] = totalScore / overlapping.size(0);
        }
        outDets.push_back(weightedDet);
    }
    // cout<<outDets<<endl;
    return;
}

void weightedNMS(const Tensor &detections, std::vector<Tensor>& outDets, int scoreDim, float minSuppressionThrs)
{
    if(detections.size(0) == 0)
        return;
    auto remaining = detections.slice(1,scoreDim, scoreDim+1).argsort(0, true).squeeze(-1);
    // cout<<remaining.sizes()<<"  "<<remaining[0];
    // cout<<detections[remaining[0]].sizes()<<"\n";
    // torch::Tensor IoUs;
    while (remaining.size(0)>0)
    {
        auto Det = detections[remaining[0]].to(torch::kCPU, false, true);
        auto firstBox = detections[remaining[0]].slice(0,0,4).unsqueeze(0);
        auto otherBoxes = detections.index(remaining).slice(1,0,4);
        // cout<<firstBox.sizes()<<"    "<<otherBoxes.sizes();
        auto IoUs = computeIoU(firstBox, otherBoxes);
        // cout<<IoUs.sizes();
        auto overlapping = remaining.index(IoUs > minSuppressionThrs);
        remaining = remaining.index(IoUs <= minSuppressionThrs);

        outDets.push_back(Det);
    }
    // cout<<outDets<<endl;
    return;
}

class NonMaxSuppression: public PipeProcessor {
public:
    float scoreClipThrs, minScoreThrs, minSuppressionThrs;
    int numKeypoints, scoreDim;
    NonMaxSuppression(int num_keypts, float clip_t=100.0, float score_t=0.8, float suppression_t=0.3, string pp_name=""): 
        PipeProcessor(2, 2, AVP_TENSOR, pp_name, STREAM_PROC), scoreClipThrs(clip_t), minScoreThrs(score_t), 
        minSuppressionThrs(suppression_t) //int dst_h, int dst_w, int obj_up_id, int obj_down_id,  dstHeight(dst_h), dstWidth(dst_w), objUpId(obj_up_id), objDownId(obj_down_id)
    {
        scoreDim = 4 + numKeypoints;
    }
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        auto detScores = in_data_list[0].tensor.clamp(-scoreClipThrs, scoreClipThrs).sigmoid();
        auto detBoxes = in_data_list[1].tensor;
        auto mask = detScores >= minScoreThrs;
        int bs = detBoxes.size(0);

        /* Attention BS must be one!!!
         * TODO: a potential solution:
         *  limit the number of output dets to a fix number,
         *  using a flag to indicate whether this placeholder has a valid det.
         */ 
    
        for(int i=0; i< bs; i++)
        {
            auto boxes = detBoxes[i].index(mask[i]);
            auto scores = detScores[i].index(mask[i]).unsqueeze(-1);

            /* NMS */
            std::vector<Tensor> outDetsList;
            weightedNMS(torch::cat({boxes, scores}, -1), outDetsList, scoreDim, minSuppressionThrs);
            int numDets = outDetsList.size();
            int i =0;
            auto outDets = torch::empty({numDets, 5}, torch::kF32);
            auto outLandMarks = torch::empty({numDets, numKeypoints}, torch::kF32);
            for(auto& det_t:outDetsList)
            {
                outDets[i].slice(0,0,4) = det_t.slice(0,0,4);
                outDets[i][4] = det_t[scoreDim];
                outLandMarks[i].slice(0,0,numKeypoints) = det_t.slice(0,4,scoreDim);
                i++;
            }
            out_data_list[0].tensor = outDets;
            out_data_list[1].tensor = outLandMarks;
        }
    }
    
};
                // outDets[i][0] = det[0] * dstHeight; // ymin
                // outDets[i][1] = det[1] * dstWidth; // xmin
                // outDets[i][2] = det[2] * dstHeight; // ymax
                // outDets[i][3] = det[3] * dstWidth; // xmax
                // outDets[i][4] = det[4+objUpId*2];
                // outDets[i][5] = det[4+objUpId*2+1]; 
                // outDets[i][6] = det[4+objDownId*2]; 
                // outDets[i][7] = det[4+objDownId*2+1];
}