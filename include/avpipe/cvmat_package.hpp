/*!
 * Deprecated
 */
#pragma once

#include <opencv2/opencv.hpp>
#include "base.hpp"

namespace avp {

using Mat = cv::Mat;

class CVMatPackage: public StreamPackage {
public:
    Mat data;
    CVMatPackage(Mat& mat_data, int mat_timestamp=-1)
    {
        data = mat_data;
        timestamp = mat_timestamp;
    }
    CVMatPackage() {timestamp = -1;}
    void* data_ptr()
    {
        return data.data;
    }
};

}