import numpy as np

from simulation.utils.load import load_raw_data
from dev_tools.run_experiments.experiment_utils import *
from simulation.utils.plot_velocity_field import save_velocity_at_first_node
from simulation.utils.plot_net_flow import save_net_flow_at_first_node
from simulation.utils.plot_pressure import save_pressure_field

desimals=3

depth = 5           # must be tree or greater
ts_per_cycle = 50

c_pial = 1.96 # (mm/s)

# Data from Bojarskaite article
vasomotion_freq = np.round(np.linspace(0.1, 0.3, 3,),desimals) # Hz
vasomotion_lambda = np.round(c_pial / vasomotion_freq,desimals) # mm
Ls_array = [0.6]

# NONREM
NON_REM_r0 = 0.006 # 6 µm
NON_REM_re = 0.013 # 13 µm
NON_REM_betas = f"{NON_REM_re/NON_REM_r0:.{desimals}f}"

NON_REM_vasomotion_eps=f"{0.001/NON_REM_r0:.{desimals}f}" 

# REM
REM_r0 = 0.0075 # 7.5 µm
REM_re = 0.0135 # 13.5 µm
REM_betas = f"{REM_re/REM_r0:.{desimals}f}"

REM_vasomotion_eps=f"{0.0005/REM_r0:.{desimals}f}"

vasomotion_lambda_str = [str(v) for v in vasomotion_lambda]
vasomotion_freq_str = [str(v) for v in vasomotion_freq]

def input_args(depth):
    inputs = [
        *make_cmd_ready("depth", depth),
        *make_cmd_ready("ts_per_cycle", ts_per_cycle),
        "--lambdas", *vasomotion_lambda_str,
        "--freq", *vasomotion_freq_str
    ]
    return inputs 

NON_REM_ARGS = [
    *make_cmd_ready("radius0", NON_REM_r0),
    *make_cmd_ready("betas", NON_REM_betas),
    *make_cmd_ready("eps", NON_REM_vasomotion_eps)]

REM_ARGS = [
    *make_cmd_ready("radius0", REM_r0),
    *make_cmd_ready("betas", REM_betas),
    *make_cmd_ready("eps", REM_vasomotion_eps)]

with ProcessPoolExecutor() as main_executor:
    future_non_rem = main_executor.submit(run_experiments, [*NON_REM_ARGS, *input_args(depth)], Ls_array)
    future_rem = main_executor.submit(run_experiments, [*REM_ARGS, *input_args(depth)], Ls_array)
    
    non_rem_paths = future_non_rem.result()
    rem_paths = future_rem.result()
    
    rem_paths_list = [item["path"] for item in rem_paths]
    non_rem_path_list = [item["path"] for item in non_rem_paths]

matched_experiments = group_matching_experiments(non_rem_path_list, rem_paths_list)

id = 0
for match in matched_experiments:
    print(f"Match on freq = {match['freq_value']}:")
    print(f"NON-REM: {match['non_rem']}")
    print(f"REM:     {match['rem']}")
    
    graph_path = match['non_rem'] 

    pressure_path_rem = f"{match['rem']}/pressure/HDF5/sols_{id}.h5"
    flux_path_rem = f"{match['rem']}/flux/HDF5/sols_{id}.h5"
    exp_data_file_path_rem = f"{match['rem']}/exp_data/exp_{id}.pkl"

    pressure_path_non_rem = f"{match['non_rem']}/pressure/HDF5/sols_{id}.h5"
    flux_path_non_rem = f"{match['non_rem']}/flux/HDF5/sols_{id}.h5"
    exp_data_file_path_non_rem = f"{match['non_rem']}/exp_data/exp_{id}.pkl"

    _, qps_non_rem, exp_data_non_rem  = load_raw_data(graph_path, pressure_path_rem, flux_path_rem, exp_data_file_path_rem)
    G, qps_rem, exp_data_rem = load_raw_data(graph_path, pressure_path_non_rem,
                                                   flux_path_non_rem, exp_data_file_path_non_rem)
    
    save_velocity_at_first_node(G, qps_rem, qps_non_rem, exp_data_rem, exp_data_non_rem, 
                                f"{match['rem']}", f"{match['non_rem']}",id)

    save_net_flow_at_first_node(G, qps_rem, qps_non_rem, exp_data_rem,exp_data_non_rem,
                                f"{match['rem']}",f"{match['non_rem']}",id)

"""
PRESSURE
need some sort of interpolation to handle areas between dots
        node_positions = np.array([G.nodes()[n]["pos"] for n in G.nodes()])
        x = node_positions[:, 0]
        y = node_positions[:, 1]

        pressure_per_node = [[sol[1](pos) for sol in qps_rem] for pos in node_positions]

        pressures_at_T = [p_list[-1] for p_list in pressure_per_node]

        plt.figure(figsize=(7, 5))

        if np.ptp(y) > 1e-8:  # if y varies, plot in 2D
            plt.scatter(x, y, c=pressures_at_T, cmap="viridis", s=80)
            plt.colorbar(label="Pressure p(x, y, T)")
            plt.xlabel("x-position", fontsize=16)
            plt.ylabel("y-position", fontsize=16)
            plt.title(f"Pressure Field at Final Time T (2D)", fontsize=16)
            plt.grid(True)
            plt.show()
        else:  # 1D line plot
            sort_idx = np.argsort(x)
            plt.plot(x[sort_idx], np.array(pressures_at_T)[sort_idx], marker="o")
            plt.xlabel("Position along 1D segment (x)", fontsize=16)
            plt.ylabel("Pressure p(x, T) dimensionless", fontsize=16)
            plt.title(f"Pressure Field at Final Time T (1D)", fontsize=16)
            plt.grid(True)
            plt.show()
"""
