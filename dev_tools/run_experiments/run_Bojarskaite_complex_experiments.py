from dev_tools.run_experiments.experiment_utils import *

desimals=3

depth = 3           # must be tree or greater
ts_per_cycle = 25

# could use these but i want to test computing them based on c/f = lambda
c_pial = 0.4 # (mm/s) Daversin-Catty 2020 se tabl II in directional flow 

# Data from Bojarskaite article
vasomotion_freq = np.round(np.linspace(0.1, 0.3, 3,),desimals) # Hz
vasomotion_lambda = np.round(c_pial / vasomotion_freq,desimals) # mm
Ls_array = np.round((vasomotion_lambda/(2*np.pi)),desimals) #mm

# NONREM
NON_REM_r0 = 0.006 # 6 µm
NON_REM_re = 0.013 # 13 µm
NON_REM_betas = f"{NON_REM_re/NON_REM_r0:.{desimals}f}"

NON_REM_vasomotion_eps=f"{0.001/NON_REM_r0:.{desimals}f}" # Lumen size from fig 2
print(NON_REM_vasomotion_eps) 
# REM
REM_r0 = 0.0075 # 7.5 µm
REM_re = 0.0135 # 13.5 µm
REM_betas = f"{REM_re/REM_r0:.{desimals}f}"

REM_vasomotion_eps=f"{0.0005/REM_r0:.{desimals}f}"# Lumen size from fig 2

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
    *make_cmd_ready("eps", NON_REM_vasomotion_eps),
    ]

REM_ARGS = [
    *make_cmd_ready("radius0", REM_r0),
    *make_cmd_ready("betas", REM_betas),
    *make_cmd_ready("eps", REM_vasomotion_eps)]

"""
with ProcessPoolExecutor() as main_executor:
    future_non_rem = main_executor.submit(run_experiments, [*NON_REM_ARGS, *input_args(depth)], Ls_array)
    future_rem = main_executor.submit(run_experiments, [*REM_ARGS, *input_args(depth)], Ls_array)
    
    non_rem_paths = future_non_rem.result()
    rem_paths = future_rem.result()

store_experiment_location({
    "Bojarskaite_NON_REM_data": non_rem_paths,
    "Bojarskaite_REM_data": rem_paths
})
"""
