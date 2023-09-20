import os
import subprocess
import time
import json
import datetime

session_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

available_gpus = [0]
# seeds = [1234, 64, 49]
seeds = [42]

with open("for_running.json", "r") as f:
    confs = json.load(f)

models = [
        # "e2e cnn 0.01ent_coef",
        # "e2e cnn 0.03ent_coef", 
        # "e2e sac cnn",
        # "e2e cnn-transformer3-fs4 0.01ent_coef",
        # "e2e sac cnn-transformer3-fs4",
        # "e2e ddpg cnn-transformer3-fs4",
        # "e2e cnn mlp 0.01ent_coef",
        # "e2e sac cnn mlp 0.01ent_coef",
        # "e2e ddpg cnn mlp",
        # "e2e ddpg cnn",
        # "e2e cnn-transformer-fs4 0.05ent_coef",
        # "e2e cnn-transformer7",
        # "e2e cnn-transformer5",
        # "e2e cnn-transformer3",
        # "multiple cnn-transformer 0.01ent_coef 8192n_steps",
        # "multiple cnn-transformer 0.01ent_coef",
        # "slate-nav-5x5"
        "slate-nav-10x10"
        # "slate-transformer-0.01ent_coef",
]
for m_name in models:
    if not m_name in confs["ocrs"].keys():
        raise ValueError(f"model {m_name} is not predefined. Please use in {confs['ocrs'].keys()}.")

envs = [
        # "cw_target",
        # "cw_push",
        # "cw_push_casual"
        # "cw_push_hard",
        # "cw_target_hard",
        # "cw_target_casual",
        # "navigation5x5"
        "navigation10x10",
        # "pushN3-hard-sparse",
        # "oooC2S2S1-hard-sparse-oc
        # "oooC2S2S1-hard-sparse",
]
for e_name in envs:
    if not e_name in confs["envs"].keys():
        raise ValueError(f"env {e_name} is not predefined. Please use in {confs['envs'].keys()}.")


# # create tmux session
# os.system(f"tmux kill-session -t {session_name}")
# os.system(f"tmux new-session -s {session_name} -d")
processes = {}

win_idx = 0
cnt = 0
for m_name in models:
    model_conf = confs["ocrs"][m_name]
    command = "python train_sb3.py "
    for key, value in model_conf.items():
        command += f"{key}={value} "
    for e_idx, e_name in enumerate(envs):
        # os.system(f"tmux new-window -t {session_name}")
        for s_idx, _seed in enumerate(seeds):
            proc_name = f'{m_name}_{e_name}_{_seed}'
            with open(f'{proc_name}.out', 'w') as f:
                dev = available_gpus[cnt % len(available_gpus)]
                additional_args = f"device={dev} "
                env_conf = confs["envs"][e_name]
                for key, value in env_conf.items():
                    additional_args += f"{key}={value} "
                # os.system(f"tmux split-window -v -p 140 -t {session_name}:{win_idx+1}")
                print(f"starting {command} {additional_args} seed={_seed}")
                proc = subprocess.Popen(f"PYTHONPATH=. {command} {additional_args} seed={_seed}", shell=True, stdout=f.fileno(), stderr=f.fileno())
                processes[proc_name] = proc 
                # os.system( f"""tmux send-keys -t {session_name}:{win_idx+1}.{s_idx+1} "{command} {additional_args} seed={_seed}" Enter"""
                # )
                cnt += 1
            # time.sleep(10)
        # os.system(f'tmux send-keys -t {session_name}:{win_idx+1}.0 "exit" Enter')
        # os.system(f"tmux select-layout -t {session_name}:{win_idx+1} even-horizontal")
        win_idx += 1

try:
    finished_processes = set()
    while len(processes.keys()) > len(finished_processes):
        time.sleep(60)
        for proc_name, proc in processes.items():
            if proc_name in finished_processes:
                continue
            status = proc.poll()
            if status == None:
                continue
            elif status == 0:
                print(f"{proc_name} finished!")
            else:
                print(f"{proc_name} failed with status {status}")
            finished_processes.add(proc_name)
finally:
    print("Cleaning processess")
    for proc in processes.values():
        proc.terminate()

print("Execution is finished")      

# os.system(f'tmux send-keys -t {session_name}:0 "exit" Enter')
