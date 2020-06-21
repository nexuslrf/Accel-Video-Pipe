from avp_automation import AVP_Automation
import argparse

parser = argparse.ArgumentParser(description="AVPipe Automation Tools")

parser.add_argument('--task_config', '-f', type=str, help="task configuration file in YAML format", required=True)
parser.add_argument('--action', '-a', type=str, default='V', 
    choices=['visualize', 'V', 'code_gen', 'G', 'profile', 'P', 'build', 'B', 'run', 'R', 'optimize', 'O'], 
    help='define the action to be done by avp_automation, the action can be one of [V, G, P, B, R, O], default is V')
parser.add_argument('--action_list', '-l', type=list, 
    help='define a sequence of actions containing [V, G, P, B, R, O], e.g., -l VGBR, when -l is set, -a option will be ignored')
parser.add_argument('--default_config', '-d', type=str, default='avp_template/default-configs.yaml', 
    help='template/default configurations of all pipeProcessors, default is avp_template/default-configs.yaml')
parser.add_argument('--num_threads', '-t', type=int, default=4, help='max number of threads will be used to optimize, default is 4')
parser.add_argument('--buffer_capacity', '-c', type=int, default=5, help='buffer capacity for asynchronous processing, default is 5')
parser.add_argument('--name', '-n', type=str, default='', help='task name of the output task, default is the name of task_config')
parser.add_argument('--out_dir', '-o', type=str, default='_output', help='output directory for the target task')
parser.add_argument('--loop_len', type=int, default=50, help='number of frames to be processed by cpp program, ' + \
                ' when loop_len<=0, it will be a while(true) loop, default is -1')
parser.add_argument('--timing', action='store_true', help='if set, the output cpp program will enable timing functions')
parser.add_argument('--hide_streams', action='store_false', help='if set, the output viz graph will not show the stream info')
args = parser.parse_args()

avp_task = AVP_Automation(args.task_config, args.default_config, args.name, args.out_dir)
if args.action_list == None:
    args.action_list = [args.action]

threads = []
for act in args.action_list:
    if act.upper().startswith('V'):
        avp_task.visualize(show_streams=args.hide_streams, threads=threads)
    elif act.upper().startswith('G'):
        avp_task.code_gen(loop_len=args.loop_len, timing=args.timing, buffer_capacity=args.buffer_capacity, threads=threads)
    elif act.upper().startswith('B'):
        avp_task.cpp_build()
    elif act.upper().startswith('R'):
        avp_task.cpp_run()
    elif act.upper().startswith('P'):
        avp_task.profile(loop_len=args.loop_len)
    elif act.upper().startswith('O'):
        threads = avp_task.multi_threading(args.num_threads)
