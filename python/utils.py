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
def gen_GV_label(cfg, horizontal=False):
    proc_label = f"{cfg['label']}:\\n{cfg['PipeProcessor']}"

    # inStreams
    inStreams_layer = ""
    for i, stream in enumerate(cfg['inStreams']):
        inStreams_layer += (f'<in{i}>'+stream['label'] + '|')
    if inStreams_layer != "":
        inStreams_layer = "{" + inStreams_layer[:-1] + "}|"
    
    # outStreams
    outStreams_layer = ""
    for i, stream in enumerate(cfg['outStreams']):
        outStreams_layer += (f'<out{i}>'+stream['label'] + '|')
    if outStreams_layer != "":
        outStreams_layer = "|{" + outStreams_layer[:-1] + "}"

    if horizontal:
        return inStreams_layer + proc_label + outStreams_layer 
    else:
        return "{" + inStreams_layer + proc_label + outStreams_layer + "}" 
     

if __name__ == "__main__":
    configs_map = default_configs_map("/Users/liangruofan1/Program/Accel-Video-Pipe/include/default-configs.yaml")
    print(configs_map['DrawDetBoxes']['args']['c'])