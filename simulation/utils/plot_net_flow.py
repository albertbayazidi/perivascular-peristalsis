import numpy as np
import simulation.utils.common_plotting_style as ps
import os

from graphnics import *
from xii import *
import matplotlib.pyplot as plt

def save_net_flow_at_first_node(G, qps_rem, qps_non_rem, exp_data_rem, exp_data_non_rem, exp_folder_rem, exp_folder_non_rem,id, plot_window=400):
    
    save_path_rem = os.path.join(exp_folder_rem,"plots","net_flow")
    save_path_non_rem = os.path.join(exp_folder_non_rem,"plots","net_flow")

    os.makedirs(save_path_rem, exist_ok=True)
    os.makedirs(save_path_non_rem, exist_ok=True)

    all_nodes = list(G.nodes())
    n0 = all_nodes[0]
    pos0 = G.nodes()[n0]["pos"]

    T_rem = exp_data_rem["T"]
    T_non_rem = exp_data_non_rem["T"]

    time_steps_rem = len(qps_rem)
    time_steps_non_rem = len(qps_non_rem)

    time_vec_rem = np.linspace(0, T_rem, time_steps_rem)
    time_vec_non_rem = np.linspace(0, T_non_rem, time_steps_non_rem)

    outflow_rem = [sol[0](pos0) for sol in qps_rem]
    outflow_non_rem = [sol[0](pos0) for sol in qps_non_rem]

    dt_rem = T_rem / time_steps_rem
    dt_non_rem = T_non_rem / time_steps_non_rem

    ys_rem = np.cumsum(outflow_rem) * dt_rem
    ys_non_rem = np.cumsum(outflow_non_rem) * dt_non_rem

    fig, ax = plt.subplots(figsize=ps.FIG_SIZE)

    ax.plot(time_vec_rem[:plot_window], ys_rem[:plot_window], **ps.STYLE_REM)
    ax.plot(time_vec_non_rem[:plot_window], ys_non_rem[:plot_window], **ps.STYLE_NON_REM)

    ax.set_title("Net flow at First Node", fontsize=16)
    ax.set_xlabel("t' [s]", fontsize=16)
    ax.set_ylabel("$\\int_0^{t'} Q'(\\tau) \\, \\mathrm{d} \\tau$ [$mm$]", fontsize=16)
    ax.grid(True)
    ax.legend()
    fig.tight_layout() 
    
    file_out_rem = os.path.join(save_path_rem,f"net_flow_at_node{str(n0)}_{str(id)}.png") 
    fig.savefig(file_out_rem, dpi=300)
    
    file_out_non_rem = os.path.join(save_path_non_rem,f"net_flow_at_node{str(n0)}_{str(id)}.png")  
    fig.savefig(file_out_non_rem, dpi=300)

    plt.close(fig)
