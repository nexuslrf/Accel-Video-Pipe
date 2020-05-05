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

if __name__ == "__main__":
    yaml_path = "/Users/liangruofan1/Program/Accel-Video-Pipe/test_python/test_yaml.yaml"
    config = read_yaml(yaml_path)
    print(config[0]['label'])