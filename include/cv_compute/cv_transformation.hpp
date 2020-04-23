#pragma once

#include "../avpipe/base.hpp"

namespace avp{

class CenterCropResize: public PipeProcessor {
public:
    int cropHeightLowBnd, cropWidthLowBnd;
    int srcHeight, srcWidth;
    int cropHeight, cropWidth;
    int dstHeight, dstWidth;
    bool flip, returnCrop;
    cv::Rect ROI;
    CenterCropResize(int src_height, int src_width, int dst_height, int dst_width, bool flip_img=false, bool return_crop=false, string pp_name=""): 
        PipeProcessor(1, 1, AVP_MAT, pp_name, STREAM_PROC), srcHeight(src_height), srcWidth(src_width), 
        dstHeight(dst_height), dstWidth(dst_width), flip(flip_img), returnCrop(return_crop)
    {
        float ratio = 1.0 * dstWidth / dstHeight;
        if (srcWidth * dstHeight > srcHeight * dstWidth)
        {
            float max_expand = srcHeight * ratio;
            cropHeightLowBnd = 0; 
            cropHeight = srcHeight;
            cropWidthLowBnd = (srcWidth - max_expand)/2;
            cropWidth = max_expand;
        }
        else
        {
            float max_expand = srcWidth / ratio;
            cropHeightLowBnd = (srcHeight - max_expand) / 2;
            cropHeight = max_expand;
            cropWidthLowBnd = 0;
            cropWidth = srcWidth;
        }
        ROI = cv::Rect(cropWidthLowBnd, cropHeightLowBnd, cropWidth, cropHeight);
        if(return_crop)
        {
            numOutStreams = 2;
        }
    }
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        for(auto &in_mat: in_data_list[0].matList)
        {
            Mat tmp_mat, crop_mat;
            crop_mat = in_mat(ROI);
            cv::resize(crop_mat, tmp_mat, {dstWidth, dstHeight}, 0, 0, cv::INTER_LINEAR);
            if(flip)
                cv::flip(tmp_mat, tmp_mat, +1);
            out_data_list[0].loadData(tmp_mat);
            if(returnCrop)
                out_data_list[1].loadData(crop_mat);
        }
    }
};

class ImgNormalization: public PipeProcessor {
    cv::Scalar mean, stdev;
public:
    ImgNormalization(cv::Scalar mean_var, cv::Scalar stdev_var, std::string pp_name=""): PipeProcessor(1,1, AVP_MAT, pp_name, STREAM_PROC),
        mean(mean_var), stdev(stdev_var)
    {} 
    ImgNormalization(float mean_var=0.5, float stdev_var=0.5, std::string pp_name=""): PipeProcessor(1,1, AVP_MAT, pp_name, STREAM_PROC)
    {
        mean = {mean_var, mean_var, mean_var};
        stdev = {stdev_var, stdev_var, stdev_var};
    } 
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        for(auto &in_mat: in_data_list[0].matList)
        {
            cv::Mat tmp_frame;
            cv::cvtColor(in_mat, tmp_frame, cv::COLOR_BGR2RGB);
            tmp_frame.convertTo(tmp_frame, CV_32F);
            tmp_frame = (tmp_frame / 255.0 - mean) / stdev;
            out_data_list[0].loadData(tmp_frame);
        }
    }
};

cv::Mat computePointAffine(cv::Mat &pointsMat, cv::Mat &affineMat, bool inverse)
{
    // cout<<pointsMat.size<<endl;
    if(!inverse)
    {
        cv::Mat ones = cv::Mat::ones(pointsMat.cols, 1, CV_32F);
        pointsMat.push_back(ones);
        return affineMat * pointsMat;
    }
    else
    {
        pointsMat.row(0)-=affineMat.at<float>(0,2);
        pointsMat.row(1)-=affineMat.at<float>(1,2);
        cv::Mat affineMatInv = affineMat(cv::Rect(0,0,2,2)).inv();
        return affineMatInv * pointsMat;
    }
}

class RotateCropResize: public PipeProcessor {
public:
    int dstHeight, dstWidth;
    int objUpIdx, objDownIdx;
    float shiftY, shiftX, boxScale;
    bool keepSquare;
    Tensor sizeScale;
    RotateCropResize(int dst_h, int dst_w, float h_scale, float w_scale, int obj_up_id=2, int obj_down_id=0, float shift_y=0.5, float shift_x=0.0,
            float box_scale=2.6, bool keep_square=true, PackType out_type=AVP_MAT, string pp_name=""): 
        PipeProcessor(3,3,out_type,pp_name,STREAM_PROC), dstHeight(dst_h), dstWidth(dst_w), objUpIdx(obj_up_id), objDownIdx(obj_down_id), 
        shiftY(shift_y), shiftX(shift_x), boxScale(box_scale), keepSquare(keep_square)
    {
        sizeScale = torch::empty({1,4}, torch::kF32);
        // float tmp_scales[4] = {h_scale, w_scale, h_scale, w_scale};
        sizeScale[0][0] = sizeScale[0][2] = h_scale;
        sizeScale[0][1] = sizeScale[0][3] = w_scale;
    }
    /*
     * in_data_list:
     *  [0]: detBoxes, [1]: detKeypoints, [2]: frameMat
     * out_data_list:
     *  [0]: cropObjs, [1]: affineMats, [2]: rotCenters
     */
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        // std::cout<<sizeScale<<"\n";
        // std::cout<<in_data_list[0].tensor()<<"\n";
        auto detBoxes = in_data_list[0].tensor() * sizeScale;
        // std::cout<<detBoxes<<"\n";
        // auto detBoxes_a = detBoxes.accessor<float, 2>(); // [ymin, xmin, ymax, xmax]
        auto frame = in_data_list[2].mat();
        int numDets = detBoxes.size(0);
        auto detKeypoint = in_data_list[1].tensor();
        auto yscales = (detBoxes.slice(1,2,3) - detBoxes.slice(1,0,1)).squeeze();
        auto xscales = (detBoxes.slice(1,3,4) - detBoxes.slice(1,1,2)).squeeze();
        // std::cout<<xscales<<" "<<yscales<<"\n";
        auto obj_Delta = (detKeypoint.slice(1,objDownIdx,objDownIdx+1) - detKeypoint.slice(1,objUpIdx,objUpIdx+1)).squeeze();
        auto angleRads = torch::atan2(obj_Delta.slice(1,0,1), obj_Delta.slice(1,1,2)).squeeze();
        // auto angleRads_a = angleRads.accessor<float, 1>();
        auto angleDegs = (angleRads * 180 / M_PI);
        auto angleDegs_a = angleDegs.accessor<float, 1>();

        auto x_centers = (detBoxes.slice(1,1,2).squeeze()/*xmin*/ + xscales * (0.5 - shiftY*torch::sin(angleRads) + 
            shiftX*torch::cos(angleRads)));
        auto x_centers_a = x_centers.accessor<float, 1>();
        auto y_centers = (detBoxes.slice(1,0,1).squeeze()/*ymin*/ + yscales * (0.5 - shiftY*torch::cos(angleRads) - 
            shiftX*torch::sin(angleRads)));
        auto y_centers_a = y_centers.accessor<float, 1>();
        if(keepSquare)
        {
            xscales = yscales = torch::max(xscales, yscales);
        }

        auto xrescales = xscales * boxScale;
        auto yrescales = yscales * boxScale;
        
        auto xrescales_a = xrescales.accessor<float, 1>();
        auto yrescales_a = yrescales.accessor<float, 1>();
        
        // cv::Mat rotCentersMat(numDets, 2, CV_32F);

        // get cropped hands 
        for(int i=0; i<numDets; i++)
        {
            auto affineMat = cv::getRotationMatrix2D(cv::Point2f(frame.cols, frame.rows)/2, -angleDegs_a[i], 1);
            affineMat.convertTo(affineMat, CV_32F);
            auto bbox = cv::RotatedRect(cv::Point2f(), frame.size(), -angleDegs_a[i]).boundingRect2f();
            affineMat.at<float>(0,2) += bbox.width/2.0 - frame.cols/2.0;
            affineMat.at<float>(1,2) += bbox.height/2.0 - frame.rows/2.0;
            cv::Mat rotFrame;
            cv::warpAffine(frame, rotFrame, affineMat, bbox.size());
            // cv::imshow("Rotated", rotFrame);
            // cv::waitKey();
            // Cropping & Point Affine Transformation
            cv::Mat_<float> pointMat(2,1, CV_32F);
            pointMat<< x_centers_a[i], y_centers_a[i];
            // cout<<pointMat<<endl;
            cv::Mat_<float> rotPtMat = computePointAffine(pointMat, affineMat, false);
            // cout<<computePointAffine(rotPtMat, affineMat, true)<<endl;
            cv::Point2f rotCenter(rotPtMat(0), rotPtMat(1));
            // Out of range cases
            float xrescale_2 = xrescales_a[i]/2, yrescale_2 = yrescales_a[i]/2;
            std::cout<<xrescale_2<<" "<<yrescale_2<<"\n";
            float xDwHalf = std::min(rotCenter.x, xrescale_2), yDwHalf = std::min(rotCenter.y, yrescale_2);
            float xUpHalf = rotCenter.x+xrescale_2 > rotFrame.cols?rotFrame.cols-rotCenter.x:xrescale_2;
            float yUpHalf = rotCenter.y+yrescale_2 > rotFrame.rows?rotFrame.rows-rotCenter.y:yrescale_2;
            auto cropObj = rotFrame(cv::Rect(rotCenter.x-xDwHalf, rotCenter.y-yDwHalf, xDwHalf+xUpHalf, yDwHalf+yUpHalf));
            cv::copyMakeBorder(cropObj, cropObj, yrescale_2-yDwHalf, yrescale_2-yUpHalf, 
                                xrescale_2-xDwHalf, xrescale_2-xUpHalf, cv::BORDER_CONSTANT);
            // return detRect(cropObj, affineMat, rotCenter);
            cv::resize(cropObj, cropObj, {dstWidth, dstHeight}, 0, 0, cv::INTER_LINEAR);

            out_data_list[0].loadData(cropObj);
            out_data_list[1].loadData(affineMat);
            out_data_list[2].loadData(rotPtMat);
        }
    }
};
}