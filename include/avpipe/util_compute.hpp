#pragma once

#include "../avpipe/base.hpp"

namespace avp {

class TemplateProcessor: public PipeProcessor {
    std::function<void(DataList&, DataList&)> runFunc_ptr;
public:
    TemplateProcessor(int num_input, int num_output, std::function<void(DataList&, DataList&)> func_ptr=NULL,
        bool skip_empty_check=false, PackType out_data_type=AVP_TENSOR, string pp_name="", PPType process_type=STREAM_PROC):
        PipeProcessor(num_input, num_output, out_data_type, pp_name, process_type), runFunc_ptr(func_ptr)
    {
        skipEmptyCheck = skip_empty_check;
    }
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        runFunc_ptr(in_data_list, out_data_list);
    }
};

class TimeUpdater: public PipeProcessor {
public:
    int timeStep;
    TimeUpdater(int num_stream, int time_step=1, PackType out_data_type=AVP_TENSOR, 
            string pp_name="", PPType process_type=STREAM_PROC): 
        PipeProcessor(num_stream, num_stream, out_data_type, pp_name, process_type), timeStep(time_step)
    {
        skipEmptyCheck = true;
    }
    void run(DataList& in_data_list, DataList& out_data_list)
    {
        for(size_t i=0; i<numInStreams; i++)
        {
            out_data_list[i] = in_data_list[i];
            out_data_list[i].timestamp+=timeStep;
        }
    }
};
/*
 * According the judging results to descide where the streams should be forwarded
 * Let's try the Lambda Function this time!
 */
// class CompoundProcessor: public PipeProcessor {
//     void (*runFunc_ptr)(DataList&, DataList&);
//     std::vector<PipeProcessor*> processorList;
// public:
//     CompoundProcessor(int num_input, int num_output, std::vector<PipeProcessor*> processor_list, void (*func_ptr)(DataList&, DataList&)=NULL,
//         PackType out_data_type=AVP_TENSOR, string pp_name="", PPType process_type=STREAM_PROC):
//         PipeProcessor(num_input, num_output, out_data_type, pp_name, process_type), runFunc_ptr(func_ptr), processorList(processor_list)
//     {}
//     void run(DataList& in_data_list, DataList& out_data_list)
//     {
//         runFunc_ptr(in_data_list, out_data_list);
//     } 
// };

}