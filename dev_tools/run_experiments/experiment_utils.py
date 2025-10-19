import numpy as np
import subprocess
from concurrent.futures import ProcessPoolExecutor
import json
import os


def store_experiment_location(all_paths):
    base_path = "results/comparison/overview.json"
    
    data = {}
    if os.path.exists(base_path):
        with open(base_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

    for name, new_results in all_paths.items():
        if name in data:
            data[name].extend(new_results)
        else:
            data[name] = new_results

    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    
    with open(base_path, "w") as f:
        json.dump(data, f, indent=4)


def make_cmd_ready(param, val):
    return f"--{param}", str(val)

def format_cmd_for_json(cmd_list):
    cmd_dict = {}
    args = cmd_list[3:]
    i = 0
    while i < len(args):
        key = args[i].lstrip('-')
        i += 1
        values = []
        while i < len(args) and not args[i].startswith('--'):
            values.append(args[i])
            i += 1
        
        if len(values) == 1:
            cmd_dict[key] = values[0]
        else:
            cmd_dict[key] = values
    return cmd_dict

def run_single_experiment(args_and_L):
    args, L = args_and_L
    cmd = ["python", "-m", "simulation.new_comparison"]
    cmd.extend(args)
    cmd.extend(["--Ls", str(L)])
    print(" ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, encoding="utf-8")
    last_line = result.stdout.strip().split('\n')[-1] if result.stdout else ""
    
    formatted_cmd = format_cmd_for_json(cmd)
    
    return {"cmd": formatted_cmd, "path": last_line}


def run_experiments(args, Ls_array):
    tasks = [(args, L) for L in Ls_array]
    results_list = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(run_single_experiment, tasks)
        results_list.extend(results)
    return results_list
