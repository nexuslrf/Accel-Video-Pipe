
import os
import sys
from textwrap import indent
from graphviz import Digraph
from utils import read_yaml, print_err, dict_has, gen_cpp_params

class AVP_Automation:
    def __init__(self, task_configs_yaml, default_configs_yaml="../include/default-configs.yaml", task_name="", out_path="_output"):
        self.task_name = task_configs_yaml.split('/')[-1][:-5] if task_name == "" else task_name
        self.out_path = os.path.join(out_path, self.task_name)
        os.makedirs(self.out_path, exist_ok=True)
        configs_list = read_yaml(default_configs_yaml)
        self.default_configs_map = dict()
        for cfg in configs_list:
            self.default_configs_map[cfg['PipeProcessor']] = cfg
        
        self.task_configs = read_yaml(task_configs_yaml)
        assert(type(self.task_configs) == list)
        self.task_components_map = dict()
        self.orderings = []
        
        for cfg in self.task_configs:
            # create empty feeding list for later use
            cfg['feeding'] = []
            # avoid repeated labels
            if cfg['label'] in self.task_components_map:
                print_err(f"{cfg['label']} has been defined!")
            # complete inStreams for task components
            default_cfg = self.default_configs_map[cfg['PipeProcessor']]
            cfg['inStreams'] = []
            cfg['outStreams'] = []
            if dict_has('binding', cfg):
                num_inStreams = len(cfg['binding'])
                for i in range(num_inStreams):
                    if i < len(default_cfg['inStreams']):
                        cfg['inStreams'].append(default_cfg['inStreams'][i])
                    else:
                        cfg['inStreams'].append({'label': f'in[{i}]'})
            else:
                cfg['binding'] = []

            self.task_components_map[cfg['label']] = cfg

        # make sure custom configs have right args
        self.configs_checking()

        for cfg in self.task_configs:
            # complete outStreams for task components
            for i, stream in enumerate(cfg['binding']):
                label = stream['label']

                idx = 0 
                if 'idx' in stream:
                    idx = stream['idx']
                else:
                    stream['idx'] = 0

                feeding_stream = {'label': cfg['label'], 'idx': i, 'out_idx': idx}
                if 'async_time' in stream and stream['async_time']:
                    feeding_stream['async_time'] = True
                self.task_components_map[label]['feeding'].append(feeding_stream)
                
                tar_components = self.task_components_map[label]
                tar_outStreams = tar_components['outStreams']
                default_outStreams = \
                    self.default_configs_map[tar_components['PipeProcessor']]['outStreams']

                while idx >= len(tar_outStreams):
                    length = len(tar_outStreams)
                    if len(default_outStreams) > length:
                        tar_outStreams.append(default_outStreams[length])
                    else:
                        tar_outStreams.append({'label': f'out[{length}]'})
                        
        self.orderings = self.topology_sort(self.task_configs)
                        
    def configs_checking(self):
        for cfg in self.task_configs:
            cfg_name = cfg['PipeProcessor']
            default_cfg = self.default_configs_map[cfg_name]
            if default_cfg['args'] is not None:
                for arg, val in default_cfg['args'].items():
                    if val == 'AVP::REQUIRED':
                        if (not dict_has('args', cfg)) or (arg not in cfg['args']):
                            print_err(f"{cfg_name}.{arg} undefined!")
            if ('args' in cfg) and (cfg['args'] is not None):
                for arg, val in cfg['args'].items():
                    if arg not in default_cfg['args']:
                        print_err(f"Unknown arg: {cfg_name}.{arg}")
    
    def topology_sort(self, thread):
        if len(thread) and type(thread[0]) == dict:
            thread = [item['label'] for item in thread]
        # analyze the topology order of input pipeProcessors
        topo_order = []
        # scan to get source node:
        # 1. no binding procs; 2. bindings are async; 3. bindings not in thread
        for label in thread:
            cfg = self.task_components_map[label]
            if len(cfg['binding']) == 0 and len(cfg['outStreams']) != 0:
                topo_order.append(label)
                continue
            # nodes can be computed in addition to source processor
            runnable = False
            for pipe in cfg['binding']:
                if (('async_time' not in pipe) or not pipe['async_time']) and pipe['label'] in thread:
                    runnable = False
                    break
                else:
                    runnable = True
            if runnable and label not in topo_order:
                topo_order.append(label)

        if len(topo_order) == 0:
            print_err("no available source PipeProcessor!")
        
        # use a topology sorting to get processor order
        pivot = 0
        while pivot<len(topo_order):
            for p in self.task_components_map[topo_order[pivot]]['feeding']:
                p_label = p['label']
                if p_label not in thread:
                    continue
                runnable  = True
                for b in self.task_components_map[p_label]['binding']:
                    if b['label'] not in topo_order and \
                        (('async_time' not in b) or not b['async_time']) and b['label'] in thread:
                        runnable = False
                        break
                if runnable and p_label not in topo_order:
                    topo_order.append(p_label)
            pivot += 1
        return topo_order

    def visualize(self, show_timing=False, show_streams=True, threads=[], show_bar=False, format='pdf'):
        def gen_gv_label(cfg, horizontal=False, show_timing=False, show_streams=True):
            if show_timing and 'timing_info' in cfg:
                proc_label = f"{cfg['label']}:\\n{cfg['PipeProcessor']}\\nTiming: {cfg['timing_info']}ms"
            else:
                proc_label = f"{cfg['label']}:\\n{cfg['PipeProcessor']}"

            if 'refs' in cfg and type(cfg['refs']) == list and len(cfg['refs']):
                refs = ""
                for label in cfg['refs']:
                    ref_cfg = self.task_components_map[label]
                    refs += f"[Ref] {label}:\\n{ref_cfg['PipeProcessor']}|"
                refs = refs[:-1] if refs != "" else ""
                proc_label += "|{" + refs + "}"
                    
            inStreams_layer = ""
            outStreams_layer = ""

            if show_streams:
                # inStreams
                for i, stream in enumerate(cfg['inStreams']):
                    inStreams_layer += (f'<in{i}>'+stream['label'] + '|')
                if inStreams_layer != "":
                    inStreams_layer = "{" + inStreams_layer[:-1] + "}|"
                
                # outStreams
                for i, stream in enumerate(cfg['outStreams']):
                    outStreams_layer += (f'<out{i}>'+stream['label'] + '|')
                if outStreams_layer != "":
                    outStreams_layer = "|{" + outStreams_layer[:-1] + "}"

            if horizontal:
                return inStreams_layer + proc_label + outStreams_layer 
            else:
                return "{" + inStreams_layer + proc_label + outStreams_layer + "}" 

        g = Digraph("AVP_"+self.task_name, format=format)
        g.attr(rankdir="TD", ratio=f'{4./3}', splines='spline')

        drawn_procs = []
        if len(threads) > 1:
            with g.subgraph(name=f'cluster_color_bar', graph_attr={'label':f'Thread Color', 'ranksep':'0.1'}) as sg:
                for i, thread in enumerate(threads):
                    for label in thread:
                        cfg = self.task_components_map[label]
                        g.node(label, gen_gv_label(cfg, show_timing=show_timing, show_streams=show_streams), 
                            shape='record', fillcolor=f'/set312/{i+1}', style='filled')
                        drawn_procs.append(label)
                    if show_bar:
                        sg.node(f'thread_{i}', f'#0{i+1}', shape='record', fillcolor=f'/set312/{i+1}', style='filled')
        
        for cfg in self.task_configs:
            if len(cfg['inStreams']) + len(cfg['outStreams']) != 0 and cfg['label'] not in drawn_procs:
                g.node(cfg['label'], gen_gv_label(cfg, show_timing=show_timing, show_streams=show_streams), 
                    shape='record')
        
        edge_list = []
        for cfg in self.task_configs:
            for i, stream in enumerate(cfg['binding']):
                line_style = 'solid'
                if 'async_time' in stream and stream['async_time']:
                    line_style = 'dashed'
                if show_streams:
                    out_idx = stream['idx']
                    g.edge(stream['label']+f':<out{out_idx}>', cfg['label']+f':<in{i}>', style=line_style)
                    # g.edge(stream['label']+f':<out{out_idx}>:s', cfg['label']+f':<in{i}>:n')
                else:
                    src_node = stream['label']; dst_node = cfg['label']
                    if (src_node, dst_node) not in edge_list:
                        g.edge(src_node, dst_node, style=line_style)
                        edge_list.append((src_node, dst_node))
                    
        g.view("AVP_"+self.task_name, self.out_path)

    def code_gen(self, additional_includes=[], loop_len=20, profile=False, threads=[], buffer_capacity=5):
        cpp_file_name = self.task_name + '.cpp'
        self.cpp_file_name = os.path.join(self.out_path, cpp_file_name)
        if len(threads) == 0:
            threads = [self.orderings]
        
        def find_thread(label):
            for i, thread in enumerate(threads):
                if label in thread:
                    return i
            return -1

        include_header = "// This cpp file is generated by AVP_Automation\n" + \
                         "// Author: RF Liang, 2020\n"
        avp_initialization = "// Initialize necessary parameters for AVP\n"
        processor_definition = "// Start processor definition\n"
        pipe_binding = "// Start pipe binding\n"
        running_avp = "// Start running pipe\n"
        profile_info = ""
        l_curly = "{"; r_curly = "}"
        default_includes = ['<iostream>', '<string>', '<vector>', '<avpipe/helper_func.hpp>']
        avp_includes = []
        
        pipe_id = 0
        out_pipe_map = dict()
        in_pipe_map = dict()
        
        for cfg in self.task_configs:
            proc_name = cfg['PipeProcessor']
            proc_label = cfg['label']
            default_cfg = self.default_configs_map[proc_name]
            # add include
            include_file = f"<{default_cfg['includeFile']}>"
            if include_file not in avp_includes:
                avp_includes.append(include_file)
            # define Processor
            def_proc = \
                f"avp::{proc_name} {proc_label}({gen_cpp_params(cfg, default_cfg)}, \"{proc_label}\");\n"
            processor_definition += def_proc
            # assigning pipe_id
            out_pipes = []
            for _ in cfg['outStreams']:
                out_pipes.append(pipe_id)
                pipe_id += 1
            out_pipe_map[proc_label] = out_pipes
        
        # include formating
        if profile:
            default_includes.append('<chrono>')
        includes = default_includes + additional_includes + avp_includes
        for include in includes:
            include_header += f"#include {include}\n"

        # do the avp initialization
        if len(threads) > 1:
            avp_initialization += f"avp::numThreads = {len(threads)};\n" + \
                f"avp::streamCapacity = {buffer_capacity};\n"

        # get in_pipes
        async_time_pipes = []
        for cfg in self.task_configs:
            cfg_label = cfg['label']
            in_pipes = []
            for bnd in cfg['binding']:
                bnd_label = bnd['label']
                bnd_id = bnd['idx'] 
                pb_id = out_pipe_map[bnd_label][bnd_id]
                if ('async_time' in bnd) and bnd['async_time']:
                    if pb_id not in async_time_pipes:
                        async_time_pipes.append(pb_id)
                in_pipes.append(pb_id)
            in_pipe_map[cfg_label] = in_pipes

        # do the stream coupling for multi-thread
        couple_stream = ""
        if len(threads) > 1:
            for thread in threads:
                for label in thread:
                    cfg = self.task_components_map[label]
                    out_pipe_to_label = [[] for _ in range(len(out_pipe_map[label]))]
                    for f_stream in cfg['feeding']:
                        in_idx = f_stream['idx']
                        out_idx = f_stream['out_idx']
                        f_label = f_stream['label']
                        out_pipe_to_label[out_idx].append((f_label, in_idx))
                    for out_idx, out_pipes in enumerate(out_pipe_to_label):
                        if len(out_pipes) > 1:
                            out_threads = dict()
                            # find the thread
                            for pipe in out_pipes:
                                t_id = find_thread(pipe[0])
                                if t_id not in out_threads:
                                    out_threads[t_id] = []
                                out_threads[t_id].append(pipe)
                            if len(out_threads) > 1:
                                couple_pipes = ""
                                tar_pipe_id = out_pipe_map[label][out_idx]
                                cnt = 0
                                for _, pipes in out_threads.items():
                                    for pipe in pipes:
                                        in_pipe_map[pipe[0]][pipe[1]] = pipe_id
                                    couple_pipes += f"&pipe[{pipe_id}], "
                                    pipe_id += 1
                                    cnt += 1
                                    if cnt == len(out_threads)-1:
                                        break
                                couple_pipes = couple_pipes[:-2] if couple_pipes != "" else ""
                                couple_stream += f"pipe[{tar_pipe_id}].coupleStream({l_curly}{couple_pipes}{r_curly});\n"

        # binding pipes
        pipe_binding += f"avp::Stream pipe[{pipe_id}];\n"
        for cfg in self.task_configs:
            label = cfg['label']
            in_pipes = ""; out_pipes = ""
            for in_idx in in_pipe_map[label]:
                in_pipes += f"&pipe[{in_idx}], "

            in_pipes = in_pipes[:-2] if in_pipes != "" else ""
            
            for out_idx in out_pipe_map[label]:
                out_pipes += f"&pipe[{out_idx}], "
            
            out_pipes = out_pipes[:-2] if out_pipes != "" else ""

            pipe_binding += f"{label}.bindStream({l_curly}{in_pipes}{r_curly}," +\
                            f" {l_curly}{out_pipes}{r_curly});\n"
        
        if couple_stream != "":
            pipe_binding += "\n// Couple streams for multi-threading\n" + \
                            couple_stream

        if len(async_time_pipes):
            pipe_binding += "\n// Fill the async_time_pipe with empty packet\n" + \
                            "// Note: work flow should not modify this empty packet!\n" + \
                            f"avp::StreamPacket nullData(avp::AVP_TENSOR, {self.orderings[0]}.timeTick);\n"
            for pb_id in async_time_pipes:
                pipe_binding += f"pipe[{pb_id}].loadPacket(nullData);\n"

        running_thread = ""
        for ti, thread in enumerate(threads):
            proc_list = ""
            for i, label in enumerate(thread):
                proc_list += f"&{label}, "
                if (i+1) % 5 == 0:
                    proc_list += "\n    "
            proc_list = proc_list.rstrip()[:-1] if proc_list != "" else proc_list
            if ti+1 == len(threads):
                if profile:
                    running_thread += f"avp::pipeThreadProcessTiming(avp::ProcList({l_curly}{proc_list}{r_curly}), {loop_len});\n" 
                else:
                    running_thread += f"avp::pipeThreadProcess(avp::ProcList({l_curly}{proc_list}{r_curly}), {loop_len});\n" 
            else:
                running_thread += f"avp_threads.push_back(std::thread(avp::pipeThreadProcess," + \
                            f" avp::ProcList({l_curly}{proc_list}{r_curly}), {loop_len}));\n" 

                       
        if len(threads) > 1:
            running_avp += "// Create threads\nstd::vector<std::thread> avp_threads;\n\n" + \
                            running_thread +\
                            "\n// Wait for joining\n" + \
                            "for(auto& thread: avp_threads)\n    thread.join();\n"
        else:
            running_avp += running_thread


        running_avp += "\nstd::cout<<\"\\nAVP finish!\\n\";"    

        # add timing info
        if profile:
            profile_info += "// Show the timing info\n"
            profile_info += "#ifdef _TIMING\n"
            profile_info += "std::cout<<\"--- AVP::TIMING_INFO::BEGIN ---\\n\";\n"
            for cfg in self.task_configs:
                cfg_label = cfg['label']
                if cfg_label in self.orderings:
                    profile_info += f"std::cout<<\"{cfg_label}: \"<<{cfg_label}.averageMeter<<\" ms\\n\";\n"
            profile_info += "std::cout<<\"--- AVP::TIMING_INFO::END ---\\n\";\n"
            profile_info += "#endif\n"

        # combining
        cpp_text = include_header + \
            f"\nint main()\n{l_curly}\n" + \
            indent(f"{avp_initialization}\n{processor_definition}\n{pipe_binding}\n{running_avp}\n{profile_info}\nreturn 0;", " "*4)+ \
            f"\n{r_curly}"

        cpp_file = open(self.cpp_file_name, "w")
        cpp_file.write(cpp_text)
        cpp_file.close()
        return

    def cmake_cpp(self, template_cmakelists = 'avp_template/template_cmakelists.txt', definitions=["_LOG_INFO"]):
        cmakelists = open(template_cmakelists, 'r').read()

        # find different chunks
        configuration_chunk = ""
        add_exectable_chunk = ""
        find = False
        for line in cmakelists.split('\n'):
            if "AVP::EXEC::BEGIN" in line:
                find = True
            if find:
                add_exectable_chunk += (line + '\n')
            else:
                configuration_chunk += (line + '\n')
            if "AVP::EXEC::END" in line:
                break
        
        configuration_chunk = configuration_chunk.replace("avp_template_project", self.task_name)
        add_exectable_chunk = add_exectable_chunk.replace("avp_template_exec", self.task_name)
        for definition in definitions:
            configuration_chunk += f"add_definitions(-D{definition})\n"
        cmakelists = configuration_chunk + '\n' + add_exectable_chunk
        fh = open(os.path.join(self.out_path, "CMakeLists.txt"), 'w')
        fh.write(cmakelists)

    def cpp_build(self, script_file=".\\avp_template\\avp_build.bat"):
        if sys.platform.startswith('win'):
            script_out = os.popen(f"{script_file} {self.out_path} {self.task_name}").read()
        else:
            script_out = os.popen(f"bash {script_file} {self.out_path} {self.task_name}").read()
        print("--- CPP Build Output ---")
        print(script_out)
        return script_out

    def cpp_run(self, script_file=".\\avp_template\\avp_run.bat"):
        if sys.platform.startswith('win'):
            script_out = os.popen(f"{script_file} {self.out_path} {self.task_name}").read()
        else:
            script_out = os.popen(f"bash {script_file} {self.out_path} {self.task_name}").read()
        print("--- CPP Run Output ---")
        print(script_out)
        return script_out

    def profile(self, loop_len=50):
        self.code_gen(loop_len=loop_len, profile=True)
        self.cmake_cpp(definitions=['_TIMING'])
        self.cpp_build()
        output = self.cpp_run()
        find = False
        for line in output.split('\n'):
            if "AVP::TIMING_INFO::END" in line:
                break

            if find:
                words = line.split()
                label = words[0][:-1]
                if label in self.orderings:
                    self.task_components_map[label]['timing_info'] = eval(words[1])

            if "AVP::TIMING_INFO::BEGIN" in line:
                find = True
        self.visualize(show_timing=True)

    def multi_threading(self, max_num_threads=2, heavy_time_thres=0):
        tmp_threads = [self.orderings]
        set_threads = []
        threads = []
        def get_thread_time(thread):
            t = 0
            for label in thread:
                t += self.task_components_map[label]['timing_info']
            return t
        
        # Note: this dependency list cannot guarantee the topology order.
        def get_dependency(label, dep_type='successors', thread=[]):
            dependency_list = [label]
            pivot = 0
            assert dep_type in ['successors', 'predecessors']
            stream_key = 'feeding' if dep_type == 'successors' else 'binding'
            while pivot < len(dependency_list):
                tmp_label = dependency_list[pivot]
                cfg = self.task_components_map[tmp_label]
                for item in cfg[stream_key]:
                    if len(thread)==0 or item['label'] in thread:
                        if 'async_time' in item and item['async_time']:
                            continue
                        if item['label'] not in dependency_list:
                            dependency_list.append(item['label'])
                pivot += 1
            if dep_type == 'predecessors':
                dependency_list = dependency_list[::-1][:-1]
            else:
                dependency_list = dependency_list[1:]
            return dependency_list

        def add_thread(thread, final=False):
            if len(thread) != 0:
                if final:
                    set_threads.append(thread)
                else:
                    tmp_threads.append(thread)

        def get_num_threads():
            return len(tmp_threads) + len(set_threads)

        total_t = get_thread_time(self.orderings)
        if heavy_time_thres <= 0:
            max_label = max(self.orderings, key=lambda x: self.task_components_map[x]['timing_info'])
            thres_t = self.task_components_map[max_label]['timing_info']
        else:
            thres_t = total_t * heavy_time_thres
        
        prev_num_threads = 0
        while max_num_threads > get_num_threads() and get_num_threads() > prev_num_threads:
            # select the most time consuming thread
            thread = max(tmp_threads, key=lambda x: get_thread_time(x))
            label = max(thread, key=lambda x: self.task_components_map[x]['timing_info'])
            prev_num_threads = get_num_threads()
            tmp_threads.remove(thread)
            if get_thread_time(thread) <= thres_t:
                add_thread(thread, True)
                continue

            # try to traverse all predecessors & successors
            predecessors = self.topology_sort(get_dependency(label, 'predecessors', thread))
            successors = self.topology_sort(get_dependency(label, 'successors', thread))
            unrelated = list(set(thread) - set(predecessors) - set(successors) - set([label]))
            unrelated.sort(key=lambda x: self.task_components_map[x]['timing_info'], reverse=True)
            thread = [label]

            if max_num_threads - get_num_threads() >= 3:
                tmp_thread = [label]

                while get_thread_time(tmp_thread) < thres_t:
                    # combining
                    thread = tmp_thread[:]

                    dep_list = []
                    # get thread adjacent dependency
                    pre_suc = False  # False: pre; True: suc
                    for t_label in thread:
                        cfg = self.task_components_map[label]
                        if not pre_suc:
                            for dep_cfg in cfg['binding']:
                                dep_label = dep_cfg['label']
                                if dep_label not in thread and (dep_label,'predecessors') not in dep_list:
                                    dep_list.append((dep_label, 'predecessors'))

                        if t_label == label:
                            pre_suc = True

                        if pre_suc:
                            for dep_cfg in cfg['feeding']:
                                dep_label = dep_cfg['label']
                                if dep_label not in thread and (dep_label, 'successors') not in dep_list:
                                    dep_list.append((dep_label, 'successors'))

                    dep_list.sort(key=lambda x: self.task_components_map[x[0]]['timing_info'])

                    candidate = None
                    for dep in dep_list:
                        if dep[1] == 'predecessors':
                            dep_suc = get_dependency(label, 'successors')
                            condition_set = set(dep_suc).intersection(set(predecessors))
                            if condition_set.issubset(set(tmp_thread)):
                                candidate = dep
                        else:
                            dep_pre = get_dependency(label, 'predecessors')
                            condition_set = set(dep_pre).intersection(set(successors))
                            if condition_set.issubset(set(tmp_thread)):
                                candidate = dep
                        
                        if candidate is not None:
                            break

                    if candidate is not None:
                        if candidate[1] == 'predecessors':
                            tmp_thread = [candidate[0]] + tmp_thread
                            predecessors.remove(candidate[0])
                        else:
                            tmp_thread = tmp_thread + [candidate[0]]
                            successors.remove(candidate[0])
                    else:
                        break
                
                # pay attention to unrelated
                for u_label in unrelated:
                    pre_t = get_thread_time(predecessors)
                    suc_t = get_thread_time(successors)
                    thd_t = get_thread_time(thread)
                    min_t = min(pre_t, suc_t, thd_t)
                    if pre_t == min_t:
                        predecessors = predecessors + [u_label]
                    elif suc_t == min_t:
                        successors = [u_label] + successors
                    else:
                        thread = thread.append(u_label)

                add_thread(predecessors)
                add_thread(thread, True)
                add_thread(successors)
                
            elif max_num_threads - get_num_threads() == 2:
                # pay attention to unrelated
                for u_label in unrelated:
                    pre_t = get_thread_time(predecessors)
                    suc_t = get_thread_time(successors)
                    thd_t = get_thread_time(thread)
                    min_t = min(pre_t, suc_t, thd_t)
                    if pre_t == min_t:
                        predecessors = predecessors + [u_label]
                    elif suc_t == min_t:
                        successors = [u_label] + successors
                    else:
                        thread = thread.append(u_label)

                # combining
                side_thread = []
                if get_thread_time(predecessors) > get_thread_time(successors):
                    thread = thread + successors
                    side_thread = predecessors
                else:
                    thread = predecessors + thread
                    side_thread = successors
                
                add_thread(thread, True)
                add_thread(side_thread)

            # there's no case when max_num_threads - len(threads) == 1

        threads = tmp_threads + set_threads
        # find the main thread and put it in the back of the threads list
        main_threads = []
        for thread in threads:
            main_procs = []
            for label in thread:
                cfg = self.default_configs_map[self.task_components_map[label]['PipeProcessor']]
                if 'mainThread' in cfg and cfg['mainThread']:
                    main_procs.append(label)
            if len(main_procs) != 0:
                main_threads.append((thread, main_procs))
        
        # remove main_threads from threads
        for thread, _ in main_threads:
            threads.remove(thread)

        if len(main_threads) > 1:
            main_thread = min(main_threads, key=lambda x: get_thread_time(x[0]))[0]
            main_threads.remove(main_thread)
            for thread, main_procs in main_threads:
                merge_procs = []
                for label in main_procs:
                    merge_procs += (get_dependency(label, 'successors', thread) + [label])
                merge_procs = list(set(merge_procs))
                for label in merge_procs:
                    thread.remove(label)
                threads.append(thread)
                main_thread += merge_procs
            main_thread = self.topology_sort(main_thread)

        elif len(main_threads) == 1:
            main_thread = main_threads[0][0]

        if len(main_threads) == 0:
            main_thread = max(threads, key=lambda x: get_thread_time(x))
            threads.remove(main_thread)
        
        threads.append(main_thread)

        return threads

if __name__ == "__main__":
    default_configs = "avp_template/default-configs.yaml"
    pose_yaml = "avp_example/pose_estimation.yaml"
    hand_yaml = "avp_example/multi_hand_tracking.yaml"
    hand_no_loop_yaml = "avp_example/multi_hand_tracking_no_loopback.yaml"
    avp_task = AVP_Automation(pose_yaml, default_configs)
    # avp_task.visualize(show_streams=False)
    avp_task.code_gen(loop_len=200)
    # avp_task.cmake_cpp()
    # avp_task.cpp_build()
    # avp_task.cpp_run()
    # avp_task.profile()
    
    threads = avp_task.multi_threading(2)
    # avp_task.visualize(show_streams=True, threads=threads, show_timing=True, format='png')
    avp_task.code_gen(loop_len=200, threads=threads)
    avp_task.cmake_cpp()
    avp_task.cpp_build()
    avp_task.cpp_run()
    print(threads)
    print("pass")