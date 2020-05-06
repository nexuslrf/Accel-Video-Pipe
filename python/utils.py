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

if __name__ == "__main__":
    configs_map = default_configs_map("/Users/liangruofan1/Program/Accel-Video-Pipe/include/default-configs.yaml")
    print(configs_map['DrawDetBoxes']['args']['c'])