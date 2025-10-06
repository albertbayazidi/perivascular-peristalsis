import pandas as pd
from simulation.utils.save_path_utils import make_experiment_folder

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


def save_raw_data(data):

    json_string, json_path = make_experiment_folder(pd.DataFrame(data))

    with open(json_path, "w") as f:
        f.write(json_string)

    return json_path
