import pandas as pd
import h5py
import numpy as np
import os
import pickle

from graphnics import *
import networkx as nx

from analytics.utils import dimensional_P
from analytics.utils import dimensional_Q

from dev_tools.run_experiments.experiment_utils import make_cmd_ready
from simulation.utils.save_path_utils import make_experiment_folder
from simulation.utils.plot_net_flow import save_net_flow_at_first_node
from simulation.utils.plot_pressure import save_pressure_field
from simulation.utils.plot_velocity_field import save_velocity_at_first_node


def make_experiment_result_dict(depth, r0, Ls, betas, experiments, Q_avg_num_results, Q_avg_analytical,G):
    data = []

    for i, exp in enumerate(experiments):
        n_cycles, ts_per_cycle, epsilon, freq, lamda = [exp[key] for key in ["n_cycles", "ts_per_cycle",
                                                                             "epsilon", "freq", "lamda",]]
        Q_avg_num_i = Q_avg_num_results[i]
        Q_avg_analytical_i = Q_avg_analytical[i]

        data.append({
                "depth": depth,
                "n_cycles": n_cycles,
                "ts_per_cycle": ts_per_cycle,
                "epsilon": epsilon,
                "freq": freq,
                "lambdas": lamda,
                "betas": betas,
                "r0": r0,
                "Ls": Ls,
                "Q_avg_num": Q_avg_num_i,
                "Q_avg_analytical": Q_avg_analytical_i,
            })

    return data

def unpack_array_as_str(arr):
    if isinstance(arr, np.ndarray):
        return [str(temp) for temp in arr.tolist()]
    return [str(arr)]

def make_experiment_command(data):
    lambdas = data["lambdas"].unique()
    freq = data["freq"].unique()
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


def save_raw_data(data, G, experiments, mu, rho):
    json_string, exp_folder = make_experiment_folder(pd.DataFrame(data))
    cmd_string = make_experiment_command(pd.DataFrame(data))

    json_path = os.path.join(exp_folder, "results.json")
    cmd_path = os.path.join(exp_folder, "command.txt")

    with open(json_path, "w") as f:
        f.write(json_string)

    with open(cmd_path, "w") as f:
        f.write(cmd_string)

    save_exp(G, experiments, exp_folder, mu, rho)

    return exp_folder, json_path


def save_exp(G, experiments, exp_folder, mu, rho):
    exp_data_folder = os.path.join(exp_folder, "exp_data")
    pressure_H5_path = os.path.join(exp_folder, "pressure/HDF5")
    pressure_pvd_path = os.path.join(exp_folder, "pressure/pvd")
    flux_H5_path = os.path.join(exp_folder, "flux/HDF5" )
    flux_pvd_path = os.path.join(exp_folder, "flux/pvd")

    os.makedirs(exp_data_folder, exist_ok=True)
    os.makedirs(pressure_H5_path, exist_ok=True)
    os.makedirs(pressure_pvd_path, exist_ok=True)
    os.makedirs(flux_H5_path, exist_ok=True)
    os.makedirs(flux_pvd_path, exist_ok=True)

    for exp_id, exp in enumerate(experiments):
        k = exp["k"]
        w = exp["w"]
        eps = exp["epsilon"]
        qps = exp["sol"]

        u, v = list(G.edges())[0]
        R0 = G.edges[u, v]["radius1"]
        
        G_nx = nx.DiGraph(G)
        for e in G_nx.edges():
            del G_nx.edges()[e]['Ainv']
            del G_nx.edges()[e]['Res']
        nx.write_gpickle(G_nx, f"{exp_folder}/G.gpickle")

        exp.pop("sol", None)
        with open(f"{exp_data_folder}/exp_{exp_id}.pkl","wb") as f:
            pickle.dump(exp,f)

        q_space = qps[0][0].function_space()
        p_space = qps[0][1].function_space()

        print("q_space", q_space)
        print("p_space", p_space)

        q_dim_func = Function(q_space, name="flux") 
        p_dim_func = Function(p_space, name="pressure")        

        pvd_q = File(f"{flux_pvd_path}/sols_{exp_id}.pvd")
        pvd_p = File(f"{pressure_pvd_path}/sols_{exp_id}.pvd")

        flux_h5_path = f"{flux_H5_path}/sols_{exp_id}.h5" 
        pressure_h5_path = f"{pressure_H5_path}/sols_{exp_id}.h5" 

        with h5py.File(flux_h5_path, "w") as h5_q, \
             h5py.File(pressure_h5_path, "w") as h5_p:

            flux_group = h5_q.create_group("flux")
            pressure_group = h5_p.create_group("pressure")

            for ix, (q, p) in enumerate(qps):
                print("iteration", ix)
                q_dim_expr = dimensional_Q(q, k, w, eps, R0)
                p_dim_expr = dimensional_P(p, k, w, eps, R0, mu, rho) 

                q_dim_func.assign(project(q_dim_expr, q_space))
                p_dim_func.assign(project(p_dim_expr, p_space))

                pvd_q << (q_dim_func, float(ix))
                pvd_p << (p_dim_func, float(ix))

                q_data = q_dim_func.vector().get_local()
                p_data = p_dim_func.vector().get_local()

                flux_group.create_dataset(f"vector_{ix}", data=q_data)
                pressure_group.create_dataset(f"vector_{ix}", data=p_data)
            
            count_q = len(q_dim_func.vector().get_local())
            count_p = len(p_dim_func.vector().get_local())

            flux_group.create_dataset("iteration", data=ix)

            print("flux V: ",count_q, "pressure M: ",count_p)

def save_plots(G, experiments, exp_folder):
    pass
    #save_net_flow_at_first_node(G, experiments, exp_folder)
    #save_pressure_field(G, experiments, exp_folder)
    #save_velocity_at_first_node(G, experiments, exp_folder)
