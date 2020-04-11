#pragma once

#include "../avpipe/base.hpp"

namespace avp{

class CenterCropResize: public PipeProcessor {
public:
    int cropHeightLowBnd, cropWidthLowBnd;
    int srcHeight, srcWidth;
    int cropHeight, cropWidth;
    int dstHeight, dstWidth;
    cv::Rect ROI;
    CenterCropResize(int src_height, int src_width, int dst_height, int dst_width, std::string pp_name=""): 
        PipeProcessor(1, 1, AVP_MAT, pp_name, STREAM_PROC), srcHeight(src_height), srcWidth(src_width), dstHeight(dst_height), dstWidth(dst_width)
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
    }
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        cv::resize(in_data_list[0].mat(ROI), out_data_list[0].mat, {dstWidth, dstHeight}, 0, 0, cv::INTER_LINEAR);
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
        cv::Mat tmp_frame;
        cv::cvtColor(in_data_list[0].mat, tmp_frame, cv::COLOR_BGR2RGB);
        tmp_frame.convertTo(tmp_frame, CV_32F);
        out_data_list[0].mat = (tmp_frame / 255.0 - mean) / stdev;
    }
};

}