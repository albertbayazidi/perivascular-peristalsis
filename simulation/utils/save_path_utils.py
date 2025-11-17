import pandas as pd
import numpy as np
import os
import hashlib
import pickle
from simulation.utils.json_serializer import to_json

def format_array(arr):
    arr = np.atleast_1d(arr)
    return [f"{x:.3e}" for x in arr]

def dataframe_to_config(df):
    row0 = df.iloc[0]

    params = {
        "depth": int(row0["depth"]),
        "epsilon": float(row0["epsilon"]),
        "betas": row0["betas"],
        "r0": float(row0["r0"]),
        "Ls": row0["Ls"],
        "n_cycles": df["n_cycles"].tolist(),
    }

    sweep_params = {
        "ts_per_cycle": df["ts_per_cycle"].tolist(),
        "freq": df["freq"].tolist(),
        "lambdas": df["lambdas"].tolist(),
        "Q_avg_num": format_array(df["Q_avg_num"]),
        "Q_avg_analytical": format_array(df["Q_avg_analytical"]),
    }

    config = {**params, "sweep": sweep_params}
    return config

def make_experiment_folder(data):
    depth = data["depth"][0]
    betas = data["betas"][0]

    if len(betas) == 1 and depth == 1:
        folder = "single_element"
    elif len(betas) == 2 and depth == 1:
        folder = "tandem_element"
    elif len(betas) == 3 and depth == 2:
        folder = "bifurcated"
    else:
        folder = "complex"

    base_path = os.path.join("results", "comparison", folder)
    config = dataframe_to_config(data)

    # Create a unique id for the experiment from parameters
    param_bytes = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
    exp_hash = hashlib.md5(param_bytes).hexdigest()[:8]
    exp_folder = os.path.join(base_path, f"exp_{exp_hash}")

    os.makedirs(exp_folder, exist_ok=True)
    json_string = to_json(config)

    return json_string, exp_folder
