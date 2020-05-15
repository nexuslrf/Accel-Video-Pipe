/*!
 * Copyright (c) 2020 by Contributors
 * author Ruofan Liang
 * helper functions for AV-Pipe including
 *  * thread processing function
 *  * glog initialization
 * @TODO in the future
 *  * cmd flags parser 
 *  * yaml parser
 */
#pragma once

#include "../avpipe/base.hpp"

namespace avp {

using ProcList = std::vector<PipeProcessor*>;

void pipeThreadProcess(ProcList processor_list, int loop_len)
{
    if(loop_len<=0)
    {
        while(1)
        {
            for(auto& proc_ptr: processor_list)
            {
                proc_ptr->process();
            }
            if(processor_list.back()->finish)
                break;
        }
    }
    else
    {
        for(int i=0; i<loop_len; i++)
        {
            for(auto& proc_ptr: processor_list)
            {
                proc_ptr->process();
            }
            if(processor_list.back()->finish)
                break;
        }
    }
}

void pipeThreadProcessTiming(ProcList processor_list, int loop_len)
{
    int cumNum = 0;
    double averageMeter=0;
    if(loop_len<=0)
    {
        while(1)
        {
            auto stop1 = std::chrono::high_resolution_clock::now();
            for(auto& proc_ptr: processor_list)
            {
                proc_ptr->process();
            }
            if(processor_list.back()->finish)
                break;
            auto stop2 = std::chrono::high_resolution_clock::now();
            auto gap = std::chrono::duration_cast<std::chrono::milliseconds>(stop2-stop1).count();
            averageMeter = ((averageMeter * cumNum) + gap) / (cumNum + 1);
            cumNum++;
        }
    }
    else
    {
        for(int i=0; i<loop_len; i++)
        {
            auto stop1 = std::chrono::high_resolution_clock::now();
            for(auto& proc_ptr: processor_list)
            {
                proc_ptr->process();
            }
            if(processor_list.back()->finish)
                break;
            auto stop2 = std::chrono::high_resolution_clock::now();
            auto gap = std::chrono::duration_cast<std::chrono::milliseconds>(stop2-stop1).count();
            averageMeter = ((averageMeter * cumNum) + gap) / (cumNum + 1);
            cumNum++;
        }
    }
    std::cout<<"AVP::AVERAGE_TIME: "<<averageMeter<<" ms\n";
}



}