import yaml

# this function will read the yaml file and output a list class
def read_yaml_all(yaml_file):
    file = open(yaml_file, 'r')
    file_data = file.read()
    file.close()

    all_configs = yaml.load_all(file_data, Loader=yaml.FullLoader)
    configs = []
    for cfg in all_configs:
        configs.append(cfg)
    return configs

def read_yaml(yaml_file):
    file = open(yaml_file, 'r')
    file_data = file.read()
    file.close()

    return yaml.load(file_data, Loader=yaml.FullLoader)

def write_yaml(cfg, yaml_file):
    file = open(yaml_file, 'w')
    yaml.dump(cfg, file)
    file.close()

def default_configs_map(default_configs_yaml="../include/default-configs.yaml"):
    configs_list = read_yaml(default_configs_yaml)
    configs_map = dict()
    for cfg in configs_list:
        configs_map[cfg['PipeProcessor']] = cfg
    return configs_map

def print_err(info):
    print(f"[AVP::Error] {info}")
    exit()

def dict_has(key, cfg):
    return (key in cfg) and (cfg[key] is not None)

# take in cfg and generate graphviz label
def gen_gv_label(cfg, horizontal=False, show_timing=False, show_streams=True):
    if show_timing and 'timing_info' in cfg:
        proc_label = f"{cfg['label']}:\\n{cfg['PipeProcessor']}\\nTiming: {cfg['timing_info']}ms"
    else:
        proc_label = f"{cfg['label']}:\\n{cfg['PipeProcessor']}"

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

def gen_cpp_params(cfg, default_cfg):
    params = ""
    # Note: after python3.6, the in-built dict is orderedDict, don't to worry about the order!
    for arg, val in default_cfg['args'].items():
        final_val = None
        if dict_has('args', cfg) and (arg in cfg['args']):
            final_val = cfg['args'][arg]
        else:
            final_val = val
        if type(final_val) in (int, float):
            params += f"{final_val}, "
        elif type(final_val) == str:
            final_val = final_val.rstrip()
            if final_val.startswith('\\`') and final_val.endswith('`'):
                params += f"{final_val[2:-1]}, "
            else:
                params += f"\"{final_val}\", "
        elif type(final_val) == list:
            c_list = "{" + f"{final_val}"[1:-1] + "}"
            params += f"{c_list}, "
        elif type(final_val) == bool:
            if final_val:
                params += "true, "
            else:
                params += "false, "

    if params != "":
        params = params[:-2]
    return params

# proc type:
#   proc: 0, src: 1, sink: 2
def get_proc_type(cfg):
    proc_type = 0
    if len(cfg['inStreams']) == 0:
        proc_type = 1
    elif len(cfg['outStreams']) == 0:
        proc_type = 2
    return proc_type

if __name__ == "__main__":
    # configs_map = default_configs_map("avp_template/default-configs.yaml")
    # print(configs_map['DrawDetBoxes']['args']['c'])
    configs_list = read_yaml('avp_example/pose_estimation.yaml')
    print(yaml.dump(configs_list))