#pragma once

#include "../avpipe/base.hpp"

namespace avp {

class TemplateProcessor: public PipeProcessor {
    void (*runFunc_ptr)(DataList&, DataList&);
public:
    TemplateProcessor(int num_input, int num_output, void (*func_ptr)(DataList&, DataList&)=NULL,
        PackType out_data_type=AVP_TENSOR, string pp_name="", PPType process_type=STREAM_PROC):
        PipeProcessor(num_input, num_output, out_data_type, pp_name, process_type), runFunc_ptr(func_ptr)
    {}
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        runFunc_ptr(in_data_list, out_data_list);
    }
};

/*
 * According the judging results to descide where the streams should be forwarded
 * Let's try the Lambda Function this time!
 */
// class Multiplexer: public PipeProcessor {
// public:
    
// };

}