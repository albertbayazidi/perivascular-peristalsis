import numpy as np

from dev_tools.run_experiments.experiment_utils import *
from dev_tools.make_plots.process_experiment_pairs import process_experiment_pair 

desimals = 3
plot_window = [0,15]
target_nodes = [0,1] #  [0, 1, 3, 7, 15, 31, 63, 123, 225, 393]

depth = 1
Ls_array = Ls_array_single = [0.6]  # depth 1
#Ls_array = Ls_array_tandem = [0.3, 0.3] # depth 1 
#Ls_array = Ls_array_bifurcated = [0.2, 0.2, 0.2] # depth 2 
#Ls_array = Ls_array_complex = [0.23]  # depth 3 
#Ls_array = Ls_array_complex = [0.23]  # depth 6
#Ls_array = Ls_array_complex = [0.13]  # depth 9

ts_per_cycle = 1000
n_cycles = 6

c_pial = 1.96 # (mm/s)

# Data from Bojarskaite article
vasomotion_freq = np.round(np.linspace(0.1, 0.3, 3,),desimals) # Hz
vasomotion_lambda = np.round(c_pial / vasomotion_freq,desimals) # mm

# NONREM
NON_REM_r0 = 0.006 # 6 µm
NON_REM_re = 0.013 # 13 µm
NON_REM_betas = np.full_like(Ls_array,float(f"{NON_REM_re/NON_REM_r0:.{desimals}f}"))

NON_REM_vasomotion_eps=f"{0.001/NON_REM_r0:.{desimals}f}" 

# REM
REM_r0 = 0.0075 # 7.5 µm
REM_re = 0.0135 # 13.5 µm
REM_betas = np.full_like(Ls_array,float(f"{REM_re/REM_r0:.{desimals}f}"))

REM_vasomotion_eps=f"{0.0005/REM_r0:.{desimals}f}"

vasomotion_lambda_str = [str(v) for v in vasomotion_lambda]
vasomotion_freq_str = [str(v) for v in vasomotion_freq]
Ls_array_str = [str(v) for v in Ls_array]
NON_REM_betas_str = [str(v) for v in NON_REM_betas]
REM_betas_str = [str(v) for v in REM_betas]


def input_args(depth):
    inputs = [
        *make_cmd_ready("depth", depth),
        *make_cmd_ready("ts_per_cycle", ts_per_cycle),
        *make_cmd_ready("n_cycles",n_cycles),
        "--Ls", *Ls_array_str, 
        "--lambdas", *vasomotion_lambda_str,
        "--freq", *vasomotion_freq_str,
    ]
    return inputs 

NON_REM_ARGS = [
    *make_cmd_ready("radius0", NON_REM_r0),
    "--betas", *NON_REM_betas_str,
    *make_cmd_ready("eps", NON_REM_vasomotion_eps)]

REM_ARGS = [
    *make_cmd_ready("radius0", REM_r0),
    "--betas", *REM_betas_str,
    *make_cmd_ready("eps", REM_vasomotion_eps)]

with ProcessPoolExecutor() as main_executor:
    future_non_rem = main_executor.submit(run_experiments, [*NON_REM_ARGS, *input_args(depth)])

    future_rem = main_executor.submit(run_experiments, [*REM_ARGS, *input_args(depth)])
    
    non_rem_paths = future_non_rem.result()
    rem_paths = future_rem.result()
    
    non_rem_path_list = [item["path"] for item in non_rem_paths]
    rem_paths_list = [item["path"] for item in rem_paths]

matched_experiments = group_matching_experiments(non_rem_path_list, rem_paths_list)

for match in matched_experiments:
    print(f"Match on freq = {match['freq_value']}:")
    print(f"NON-REM: {match['non_rem']}")
    print(f"REM:     {match['rem']}")
    
    process_experiment_pair(match['non_rem'],match['rem'],plot_window,target_nodes)
