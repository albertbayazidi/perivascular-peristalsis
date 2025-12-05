import numpy as np
import simulation.utils.common_plotting_style as ps
import os

from graphnics import *
from xii import *
import matplotlib.pyplot as plt

def save_net_flow_at_nodes(G, qps_non_rem, qps_rem,  exp_data_non_rem, exp_data_rem, 
                           exp_folder_non_rem, exp_folder_rem,
                           target_node_indices=None, plot_window=-1):

    target_t_start = None
    target_t_end = None
    if plot_window == -1:
        pass 
    elif isinstance(plot_window, (list, tuple)):
        target_t_start, target_t_end = plot_window
    else:
        target_t_start, target_t_end = 0, plot_window

    save_path_rem = os.path.join(exp_folder_rem, "plots", "net_flow")
    save_path_non_rem = os.path.join(exp_folder_non_rem, "plots", "net_flow")

    os.makedirs(save_path_rem, exist_ok=True)
    os.makedirs(save_path_non_rem, exist_ok=True)

    all_nodes = list(G.nodes())

    if target_node_indices is None:
        target_node_indices = [0]

    T_rem = exp_data_rem["T"]
    T_non_rem = exp_data_non_rem["T"]

    time_steps_rem = len(qps_rem)
    time_steps_non_rem = len(qps_non_rem)

    time_vec_rem = np.linspace(0, T_rem, time_steps_rem)
    time_vec_non_rem = np.linspace(0, T_non_rem, time_steps_non_rem)
    
    dt_rem = T_rem / time_steps_rem
    dt_non_rem = T_non_rem / time_steps_non_rem

    idx_start_nr, idx_end_nr = ps.get_time_indices(time_vec_non_rem, target_t_start, target_t_end)
    idx_start_r, idx_end_r = ps.get_time_indices(time_vec_rem, target_t_start, target_t_end)

    for node_idx in target_node_indices:
        
        if node_idx >= len(all_nodes):
            print(f"Warning: Node index {node_idx} out of bounds. Skipping.")
            continue

        current_node_id = all_nodes[node_idx]
        pos_curr = G.nodes()[current_node_id]["pos"]

        outflow_rem = [sol[0](pos_curr) for sol in qps_rem]
        outflow_non_rem = [sol[0](pos_curr) for sol in qps_non_rem]

        ys_rem = np.cumsum(outflow_rem) * dt_rem
        ys_non_rem = np.cumsum(outflow_non_rem) * dt_non_rem
        
        ys_non_rem_sliced = ys_non_rem[idx_start_nr:idx_end_nr]
        time_vec_non_rem_sliced = time_vec_non_rem[idx_start_nr:idx_end_nr]
        
        ys_rem_sliced = ys_rem[idx_start_r:idx_end_r]
        time_vec_rem_sliced = time_vec_rem[idx_start_r:idx_end_r]

        # Plotting
        fig, ax = plt.subplots(figsize=ps.FIG_SIZE)

        ax.plot(time_vec_rem_sliced, ys_rem_sliced, **ps.STYLE_REM)
        ax.plot(time_vec_non_rem_sliced, ys_non_rem_sliced, **ps.STYLE_NON_REM)

        ax.set_title(f"Net flow at Node {node_idx}", fontsize=16)
        ax.set_xlabel("t' [sec]", fontsize=16)
        ax.set_ylabel("$\\int_0^{t'} Q'(\\tau) \\, \\mathrm{d} \\tau$ [$mm^3$]", fontsize=16)
        ax.grid(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True)
        fig.tight_layout() 
        
        # Unique filename using node index
        filename = f"net_flow_at_node_{str(node_idx)}.png"
        
        file_out_rem = os.path.join(save_path_rem, filename) 
        fig.savefig(file_out_rem, dpi=300)
        
        file_out_non_rem = os.path.join(save_path_non_rem, filename)  
        fig.savefig(file_out_non_rem, dpi=300)

        plt.close(fig)
