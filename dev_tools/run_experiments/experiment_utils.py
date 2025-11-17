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
        key = args[i].lstrip("-")
        i += 1
        values = []
        while i < len(args) and not args[i].startswith("--"):
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
    last_line = result.stdout.strip().split("\n")[-1] if result.stdout else ""

    formatted_cmd = format_cmd_for_json(cmd)
    
    return {"cmd": formatted_cmd, "path": last_line}

def get_freq_key(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    freq_key = str(data["sweep"]["freq"][0])
    return freq_key
        
def group_matching_experiments(non_rem_path_list, rem_path_list):
    non_rem_lookup = {}
    rem_lookup = {}

    for path in non_rem_path_list:
        freq_key = get_freq_key(path)
        if freq_key in non_rem_lookup:
            print(f"Warning: Duplicate Ls key {freq_key} in NON-REM list.")
        non_rem_lookup[freq_key] = path

    for path in rem_path_list:
        freq_key = get_freq_key(path)
        if freq_key in rem_lookup:
            print(f"Warning: Duplicate Ls key {freq_key} in REM list.")
        rem_lookup[freq_key] = path

    matched_pairs = []
    for freq_key, non_rem_path in non_rem_lookup.items():
        if freq_key in rem_lookup:
            rem_path = rem_lookup[freq_key]

            matched_pairs.append({
                "freq_value": freq_key,
                "non_rem": os.path.dirname(non_rem_path),
                "rem": os.path.dirname(rem_path)
            })

    return matched_pairs

def run_experiments(args, Ls_array):
    freqs = args[-3:]
    lambdas = args[-7:-4]
    results_list = []

    modified_tasks = []
    for lam, freq in zip(lambdas, freqs):
        temp_tasks = []
        for item in args:
            if item == "--lambdas":
                temp_tasks.append("--lambdas")
                temp_tasks.append(lam)
                temp_tasks.append("--freq")
                temp_tasks.append(freq)
                break
            else:
                temp_tasks.append(item)
        modified_tasks.append(temp_tasks)

    tot_tasks = [(task, L) for task in modified_tasks for L in Ls_array]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_single_experiment, tot_tasks))
        results_list.extend(results)
    return results_list
