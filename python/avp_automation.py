
import utils
from graphviz import Digraph

class AVP_Automation:
    def __init__(self, task_configs_yaml, default_configs_yaml="../include/default-configs.yaml", task_name=""):
        self.task_name = task_configs_yaml.split('/')[-1][:-5] if task_name == "" else task_name
        configs_list = utils.read_yaml(default_configs_yaml)
        self.default_configs_map = dict()
        for cfg in configs_list:
            self.default_configs_map[cfg['PipeProcessor']] = cfg
        
        self.task_configs = utils.read_yaml(task_configs_yaml)
        assert(type(self.task_configs) == list)
        self.task_components_map = dict()
        
        for cfg in self.task_configs:
            # avoid repeated labels
            if cfg['label'] in self.task_components_map:
                utils.print_err(f"{cfg['label']} has been defined!")
            # complete inStreams for task components
            default_cfg = self.default_configs_map[cfg['PipeProcessor']]
            cfg['inStreams'] = []
            cfg['outStreams'] = []
            if utils.dict_has('binding', cfg):
                num_inStreams = len(cfg['binding'])
                for i in range(num_inStreams):
                    if i < len(default_cfg['inStreams']):
                        cfg['inStreams'].append(default_cfg['inStreams'][i])
                    else:
                        cfg['inStreams'].append({'label': f'in[{i}]'})
                
            self.task_components_map[cfg['label']] = cfg

        # make sure custom configs have right args
        self.configs_checking()

        for cfg in self.task_configs:
            # complete outStreams for task components
            if utils.dict_has('binding', cfg):
                for stream in cfg['binding']:
                    label = stream['label']
                    idx = 0 if 'idx' not in stream else stream['idx']
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
                        
    def configs_checking(self):
        for cfg in self.task_configs:
            cfg_name = cfg['PipeProcessor']
            default_cfg = self.default_configs_map[cfg_name]
            if default_cfg['args'] is not None:
                for arg, val in default_cfg['args'].items():
                    if val == 'AVP::REQUIRED':
                        if (not utils.dict_has('args', cfg)) or (arg not in cfg['args']):
                            utils.print_err(f"{cfg_name}.{arg} undefined!")
            if ('args' in cfg) and (cfg['args'] is not None):
                for arg, val in cfg['args'].items():
                    if arg not in default_cfg['args']:
                        utils.print_err(f"Unknown arg: {cfg_name}.{arg}")

    def visualize(self, out_path='./'):
        g = Digraph("AVP_"+self.task_name)
        g.attr(rankdir="TD")
        for cfg in self.task_configs:
            g.node(cfg['label'], utils.gen_GV_label(cfg), shape='record')
        for cfg in self.task_configs:
            if utils.dict_has('binding', cfg):
                for i, stream in enumerate(cfg['binding']):
                    out_idx = 0 if 'idx' not in stream else stream['idx']
                    g.edge(stream['label']+f':<out{out_idx}>:s', cfg['label']+f':<in{i}>:n')
        g.view("AVP_"+self.task_name, out_path)

    def code_gen(self):
        pass

    def profile(self):
        pass

    def optimize(self):
        pass

if __name__ == "__main__":
    root_dir = "/Users/liangruofan1/Program/Accel-Video-Pipe/"
    default_configs = root_dir + "include/default-configs.yaml"
    pose_yaml = root_dir + "avp_example/pose_estimation.yaml"
    hand_yaml = root_dir + "avp_example/multi_hand_tracking.yaml"
    avp_task = AVP_Automation(pose_yaml, default_configs)
    avp_task.visualize()
    print("pass")