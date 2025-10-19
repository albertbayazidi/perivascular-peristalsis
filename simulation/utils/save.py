import pandas as pd
from simulation.utils.save_path_utils import make_experiment_folder
from dev_tools.run_experiments.experiment_utils import *

def make_experiment_result_dict(depth,r0, Ls, betas, experiments, Q_avg_num_results, Q_avg_analytical):
    data = []

    for i, exp in enumerate(experiments):
        n_cycles, ts_per_cycle, epsilon, freq, lamda = [exp[key] for key in ["n_cycles", "ts_per_cycle",
                                                                             "epsilon", "freq", "lamda",]]
        Q_avg_num_i = Q_avg_num_results[i]
        Q_avg_analytical_i =  Q_avg_analytical[i]

        data.append({
            "depth":depth, 
            "n_cycles": n_cycles,
            "ts_per_cycle": ts_per_cycle,
            "epsilon": epsilon,
            "freq": freq,
            "lambdas": lamda,
            "betas": betas,
            "r0": r0,
            "Ls": Ls,
            "Q_avg_num": Q_avg_num_i,
            "Q_avg_analytical": Q_avg_analytical_i
        }) 

    return data

def unpack_array_as_str(arr):
    if isinstance(arr, np.ndarray):
        return [str(temp) for temp in arr.tolist()]
    return [str(arr)]

def make_experiment_command(data):
    lambdas =data["lambdas"].unique() 
    freq =data["freq"].unique() 
    ts_per_cycle = data["ts_per_cycle"].unique()
    n_cycles = data["n_cycles"].unique()

    data = data.iloc[0]

    depth = data["depth"].item()
    radius0 = data["r0"].item()
    epsilon = data["epsilon"].item()

    if depth > 2:
        betas = data["betas"][0]
        Ls = data["Ls"][0]
    else:
        betas = data["betas"]
        Ls = data["Ls"]

    inputs = [
        *make_cmd_ready("eps", epsilon),
        *make_cmd_ready("radius0", radius0),
        *make_cmd_ready("depth", depth),
        "--ts_per_cycle",*unpack_array_as_str(ts_per_cycle),
        "--n_cycles",*unpack_array_as_str(n_cycles),
        "--betas",*unpack_array_as_str(betas),
        "--Ls",*unpack_array_as_str(Ls),
        "--lambdas",*unpack_array_as_str(lambdas),
        "--freq",*unpack_array_as_str(freq),
    ]

    cmd = ["python", "-m", "simulation.new_comparison"]
    cmd.extend(inputs)

    return " ".join(cmd) 

def save_raw_data(data):

    json_string, exp_folder = make_experiment_folder(pd.DataFrame(data))
    cmd_string = make_experiment_command(pd.DataFrame(data))
    
    json_path = os.path.join(exp_folder, "results.json")
    cmd_path = os.path.join(exp_folder, "command.txt") 

    with open(json_path, "w") as f:
        f.write(json_string)
    
    with open(cmd_path, "w") as f:
        f.write( cmd_string)


    return json_path
