
import utils

class avp_configs:
    def __init__(self, task_configs_yaml, default_configs_yaml="../include/default-configs.yaml"):
        configs_list = read_yaml(default_configs_yaml)
        self.default_configs_map = dict()
        for cfg in configs_list:
            self.default_configs_map[cfg['PipeProcessor']] = cfg
        
        self.task_configs = utils.read_yaml(task_configs_yaml)
        assert(type(self.task_configs) == list)

        utils.configs_checking(self.task_configs, self.default_configs_map)

    def visualization(self):
        pass

    def code_gen(self):
        pass

    def profiling(self):
        pass

    def optimization(self):
        pass