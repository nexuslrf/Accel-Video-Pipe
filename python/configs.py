
import utils

class AVP_Configs:
    def __init__(self, task_configs_yaml, default_configs_yaml="../include/default-configs.yaml"):
        configs_list = utils.read_yaml(default_configs_yaml)
        self.default_configs_map = dict()
        for cfg in configs_list:
            self.default_configs_map[cfg['PipeProcessor']] = cfg
        
        self.task_configs = utils.read_yaml(task_configs_yaml)
        assert(type(self.task_configs) == list)
        self.task_components_map = dict()
        for cfg in self.task_configs:
            if cfg['label'] in self.task_components_map:
                utils.print_err(f"{cfg['label']} has been defined!")
            self.task_components_map[cfg['label']] = cfg

        self.configs_checking()
    
    # make sure custom configs have right args
    def configs_checking(self):
        for cfg in self.task_configs:
            cfg_name = cfg['PipeProcessor']
            default_cfg = self.default_configs_map[cfg_name]
            if default_cfg['args'] is not None:
                for arg, val in default_cfg['args'].items():
                    if val == 'AVP::REQUIRED':
                        if ('args' not in cfg) or (cfg['args'] is None) or (arg not in cfg['args']):
                            utils.print_err(f"{cfg_name}.{arg} undefined!")
            if ('args' in cfg) and (cfg['args'] is not None):
                for arg, val in cfg['args'].items():
                    if arg not in default_cfg['args']:
                        utils.print_err(f"Unknown arg: {cfg_name}.{arg}")

    def visualization(self):
        pass

    def code_gen(self):
        pass

    def profiling(self):
        pass

    def optimization(self):
        pass

if __name__ == "__main__":
    root_dir = "/Users/liangruofan1/Program/Accel-Video-Pipe/"
    default_configs = root_dir + "include/default-configs.yaml"
    pose_yaml = root_dir + "avp_example/pose_estimation.yaml"
    hand_yaml = root_dir + "avp_example/multi_hand_tracking.yaml"
    AVP_Configs(hand_yaml, default_configs)
    print("pass")