
import os
from textwrap import indent
from graphviz import Digraph
from utils import read_yaml, print_err, dict_has, gen_gv_label, gen_cpp_params

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
                self.task_components_map[label]['feeding'].append({'label': cfg['label'], 'idx': i})
                idx = 0 
                if 'idx' in stream:
                    idx = stream['idx']
                else:
                    stream['idx'] = 0
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

        # analyze the topology order of input pipeProcessors
        for cfg in self.task_configs:
            cfg_label = cfg['label']
            if len(cfg['binding']) == 0 and len(cfg['outStreams']) != 0:
                self.orderings.append(cfg_label)
        
        if len(self.orderings) == 0:
            print_err("no available source PipeProcessor!")
        
        # nodes can be computed in addition to source processor
        for cfg in self.task_configs:
            runnable = False
            for pipe in cfg['binding']:
                if ('async_time' not in pipe) or not pipe['async_time']:
                    runnable = False
                    break
                else:
                    runnable = True
            if runnable and (cfg['label'] in self.orderings):
                self.orderings.append(cfg['label'])
        
        # use a topology sorting to get processor order
        pivot = 0
        while pivot<len(self.orderings):
            for p in self.task_components_map[self.orderings[pivot]]['feeding']:
                p_label = p['label']
                runnable  = True
                for b in self.task_components_map[p_label]['binding']:
                    if b['label'] not in self.orderings and \
                        (('async_time' not in b) or not b['async_time']):
                        runnable = False
                        break
                if runnable and p_label not in self.orderings:
                    self.orderings.append(p_label)
            pivot += 1
                        
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

    def visualize(self, show_timing=False):
        g = Digraph("AVP_"+self.task_name)
        g.attr(rankdir="TD", ratio=f'{4./3}', splines='spline')
        for cfg in self.task_configs:
            g.node(cfg['label'], gen_gv_label(cfg, show_timing=show_timing), shape='record')
        for cfg in self.task_configs:
            for i, stream in enumerate(cfg['binding']):
                out_idx = stream['idx']
                g.edge(stream['label']+f':<out{out_idx}>', cfg['label']+f':<in{i}>')
                # g.edge(stream['label']+f':<out{out_idx}>:s', cfg['label']+f':<in{i}>:n')
        g.view("AVP_"+self.task_name, self.out_path)

    def code_gen(self, additional_includes=[], loop_len=20, pass_info=False, profile=False):
        cpp_file_name = self.task_name + '.cpp'
        self.cpp_file_name = os.path.join(self.out_path, cpp_file_name)

        include_header = "// This cpp file is generated by AVP_Automation\n" + \
                         "// Author: RF Liang, 2020\n"
        processor_definition = "// Start processor definition\n"
        pipe_binding = "// Start pipe binding\n"
        running_pipe = "// Start running pipe\n"
        profile_info = ""
        l_curly = "{"; r_curly = "}"
        default_includes = ['<iostream>', '<string>', '<vector>']
        avp_includes = []
        
        pipe_id = 0
        pipe_map = dict()
        
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
            stream_id_map = []
            for _ in cfg['outStreams']:
                stream_id_map.append(pipe_id)
                pipe_id += 1
            pipe_map[proc_label] = stream_id_map
        
        # include formating
        if profile:
            default_includes.append('<chrono>')
        includes = default_includes + additional_includes + avp_includes
        for include in includes:
            include_header += f"#include {include}\n"

        # binding pipes
        async_time_pipes = []
        pipe_binding += f"avp::Stream pipe[{pipe_id}];\n"
        for cfg in self.task_configs:
            cfg_label = cfg['label']
            in_pipes = ""; out_pipes = ""
            for bnd in cfg['binding']:
                bnd_label = bnd['label']
                bnd_id = bnd['idx'] 
                pb_id = pipe_map[bnd_label][bnd_id]
                in_pipes += f"&pipe[{pb_id}], "
                if ('async_time' in bnd) and bnd['async_time']:
                    if pb_id not in async_time_pipes:
                        async_time_pipes.append(pb_id)

            if in_pipes != "":
                in_pipes = in_pipes[:-2]
            
            for i in range(len(cfg['outStreams'])):
                out_pipes += f"&pipe[{pipe_map[cfg_label][i]}], "
            if out_pipes != "":
                out_pipes = out_pipes[:-2]

            pipe_binding += f"{cfg_label}.bindStream({l_curly}{in_pipes}{r_curly}," +\
                            f" {l_curly}{out_pipes}{r_curly});\n"
        
        if len(async_time_pipes):
            pipe_binding += "\n// Fill the async_time_pipe with empty packet\n" + \
                            "// Note: work flow should not modify this empty packet!\n" + \
                            f"avp::StreamPacket nullData(avp::AVP_TENSOR, {self.orderings[0]}.timeTick);\n"
            for pb_id in async_time_pipes:
                pipe_binding += f"pipe[{pb_id}].loadPacket(nullData);\n"

        # running pipes
        # add process!
        for p_label in self.orderings:
            running_pipe += f"{p_label}.process();\n"
            if pass_info:
                running_pipe += f"std::cout<<\"{p_label} pass!\\n\";\n"
        
        if profile:
            running_pipe = "auto stop1 = std::chrono::high_resolution_clock::now();\n" + running_pipe +\
                "auto stop2 = std::chrono::high_resolution_clock::now();\n" +\
                "auto gap = std::chrono::duration_cast<std::chrono::milliseconds>(stop2-stop1).count();\n" +\
                "averageMeter = ((averageMeter * cumNum) + gap) / (cumNum + 1);\ncumNum++;\n"
                
        indent_running_pipe = indent(running_pipe, " "*4)
        
        if loop_len >= 0:
            running_pipe = f"for(int i=0; i<{loop_len}; i++)\n{l_curly}\n{indent_running_pipe}{r_curly}\n"
        else:
            running_pipe = f"while(1)\n{l_curly}\n{indent_running_pipe}{r_curly}\n"
        if profile:
            running_pipe = "int cumNum = 0;\ndouble averageMeter=0;\n\n" + running_pipe
        running_pipe += "std::cout<<\"\\nAVP cpp test pass!\\n\";"    

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
            profile_info += "std::cout<<\"AVP::AVERAGE_TIME: \"<<averageMeter<<\" ms\\n\";"

        # combining
        cpp_text = include_header + \
            f"\nint main()\n{l_curly}\n" + \
            indent(f"{(processor_definition)}\n{pipe_binding}\n{running_pipe}\n{profile_info}\nreturn 0;", " "*4)+ \
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

    def cpp_build(self, script_file="avp_template/avp_build.sh"):
        script_out = os.popen(f"bash {script_file} {self.out_path} {self.task_name}").read()
        print("--- CPP Build Output ---")
        print(script_out)
        return script_out

    def cpp_run(self, script_file="avp_template/avp_run.sh"):
        script_out = os.popen(f"bash {script_file} {self.out_path} {self.task_name}").read()
        print("--- CPP Run Output ---")
        print(script_out)
        return script_out

    def profile(self, loop_len=50):
        self.code_gen(loop_len=loop_len, profile=True, pass_info=False)
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
            predecessors = get_dependency(label, 'predecessors', thread)
            successors = get_dependency(label, 'successors', thread)
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
        return threads

    def optimize(self):
        pass

if __name__ == "__main__":
    root_dir = "/Users/liangruofan1/Program/Accel-Video-Pipe/"
    default_configs = root_dir + "avp_template/default-configs.yaml"
    pose_yaml = root_dir + "avp_example/pose_estimation.yaml"
    hand_yaml = root_dir + "avp_example/multi_hand_tracking.yaml"
    hand_no_loop_yaml = root_dir + "avp_example/multi_hand_tracking_no_loopback.yaml"
    avp_task = AVP_Automation(hand_no_loop_yaml, default_configs)
    # avp_task.visualize()
    # avp_task.code_gen(loop_len=200)
    # avp_task.cmake_cpp()
    # avp_task.cpp_build()
    # avp_task.profile()
    # --- For Dev ---
    # timing_info = \
    # [
    #     ('videoSrc', 2.06),
    #     ('crop', 0.16),
    #     ('normalization', 0.62),
    #     ('matToTensor', 0.32),
    #     ('CNN', 80.38),
    #     ('filter', 2.84),
    #     ('maxPred', 0.18),
    #     ('getKeypoint', 0.02),
    #     ('draw', 0.04),
    #     ('imshow', 23.28)
    # ]
    timing_info = \
        [
            ('videoSrc', 11.54),
            ('crop', 0),
            ('normalization', 0.26),
            ('matToTensor', 0.06),
            ('PalmCNN', 37.24),
            ('decodeBoxes', 0.1),
            ('NMS', 0.06),
            ('palmRotateCropResize', 4.55102),
            ('normalization2', 0.122449),
            ('multiCropToTensor', 0),
            ('HandCNN', 33.8776),
            ('rotateBack', 0),
            ('drawKeypoint', 0),
            ('imshow_kp', 22.82),
            ('drawDet', 0)
        ]
    for label, t in timing_info:
        avp_task.task_components_map[label]['timing_info'] = t
    
    threads = avp_task.multi_threading(4)
    print(threads)
    # --- ---
    print("pass")