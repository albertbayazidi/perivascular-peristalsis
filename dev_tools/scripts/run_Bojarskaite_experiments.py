import numpy as np
import subprocess

desimals=3
def make_cmd_ready(param_name, arr):
    return f"--{param_name} " + " ".join(str(val) for val in arr)

depth = "--depth 2"
ts_per_cycle = "--ts_per_cycle 10"

# could use these but i want to test computing them based on c/f = lambda
c_pial = 0.4 # Daversin-Catty 2020 se tabl II in directional flow 

# Data from Bojarskaite article
vasomotion_freq = np.round(np.linspace(0.1, 0.3, 3, endpoint=False),desimals) 
vasomotion_lambda = np.round(c_pial / vasomotion_freq,desimals)
Ls_array = np.round((vasomotion_lambda/2*np.pi),desimals)

# NONREM
NON_REM_r0 = "--radius0 0.06" # 60 µm
NON_REM_re = "--radius0 0.13" # 130 µm
NON_REM_betas = f"--betas {0.13/0.6:.{desimals}f}"

NON_REM_vasomotion_eps=f"--eps {0.001/0.07:.{desimals}f}" # Lumen size from fig 2, will be replaced with data from datasett

# REM
REM_r0 = "--radius0 0.07" # 70 µm
REM_re = "--radius0 0.12" # 120 µm
REM_betas = f"--betas {0.12/0.7:.{desimals}f}"

REM_vasomotion_eps=f"--eps {0.0005/0.07:.{desimals}f}"# Lumen size from fig 2, will be replaced with data from datasett 

# RUN NONREM
cmd = []
for L in Ls_array:
    cmd = [
        "python", "-m", "simulation.new_comparison",
        NON_REM_vasomotion_eps, NON_REM_r0, depth, NON_REM_betas,
        make_cmd_ready("lambdas",vasomotion_lambda),
        make_cmd_ready("freq",vasomotion_freq),
        ts_per_cycle,"--Ls " + str(L)   
    ]
    
    print(f"Executing: {' '.join(cmd)}")
subprocess.Popen(cmd)

# RUN REM
cmd = []
for L in Ls_array:
    cmd = [
        "python", "-m", "simulation.new_comparison",
        REM_vasomotion_eps, REM_r0, depth, REM_betas,
        make_cmd_ready("lambdas",vasomotion_lambda),
        make_cmd_ready("freq",vasomotion_freq),
        ts_per_cycle,"--Ls " + str(L)   
    ]
    
    print(f"Executing: {' '.join(cmd)}")
subprocess.Popen(cmd)

