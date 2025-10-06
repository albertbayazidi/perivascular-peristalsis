import numpy as np
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import json
import os

def store_experiment_location(all_paths):
    base_path = "results/comparison/overview.json"
    
    if os.path.exists(base_path):
        with open(base_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    for name, path in all_paths.items():
        if name in data:
            if path not in data[name]:
                data[name].append(path)
        else:
            data[name] = [path]

    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    
    with open(base_path, "w") as f: # could have to change with my own serilizaer
        json.dump(data, f, indent=4)


def make_cmd_ready(param,val):
    return f"--{param}",str(val)


def run_single_experiment(args_and_L):
    args, L = args_and_L
    cmd = ["python", "-m", "simulation.new_comparison"]
    cmd.extend(args)
    cmd.extend(["--Ls", str(L)])
    print(" ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, encoding="utf-8")
    last_line = result.stdout.strip().split('\n')[-1] if result.stdout else ""
    return L, last_line

def run_experiments(args, Ls_array):
    tasks = [(args, L) for L in Ls_array]
    json_dict = {}
    with ProcessPoolExecutor() as executor:
        for L, json_path in executor.map(run_single_experiment, tasks):
            json_dict[L] = json_path
    return json_dict

