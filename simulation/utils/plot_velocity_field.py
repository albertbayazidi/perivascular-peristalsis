import os
import numpy as np
import matplotlib.pyplot as plt
import simulation.utils.common_plotting_style as ps

import numpy as np
import simulation.utils.common_plotting_style as ps
import os
import matplotlib.pyplot as plt

def save_velocity_at_nodes(G, qps_non_rem, qps_rem, exp_data_non_rem, exp_data_rem, 
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

    save_path_rem = os.path.join(exp_folder_rem, "plots", "velocity")
    save_path_non_rem = os.path.join(exp_folder_non_rem, "plots", "velocity")

    os.makedirs(save_path_rem, exist_ok=True)
    os.makedirs(save_path_non_rem, exist_ok=True)

    for e in G.edges():
        radius1 = G.edges()[e]["radius1"]
        radius2 = G.edges()[e]["radius2"]
        # Area of annulus = pi * (R_outer^2 - R_inner^2)
        G.edges()[e]["area"] = np.pi * (radius2**2 - radius1**2)

    all_nodes = list(G.nodes())

    # Default to the first node if no list is provided
    if target_node_indices is None:
        target_node_indices = [0]

    T_non_rem = exp_data_non_rem["T"]
    time_steps_non_rem = len(qps_non_rem)
    time_vec_non_rem = np.linspace(0, T_non_rem, time_steps_non_rem)

    T_rem = exp_data_rem["T"]
    time_steps_rem = len(qps_rem)
    time_vec_rem = np.linspace(0, T_rem, time_steps_rem)

    idx_start_nr, idx_end_nr = ps.get_time_indices(time_vec_non_rem, target_t_start, target_t_end)
    idx_start_r, idx_end_r = ps.get_time_indices(time_vec_rem, target_t_start, target_t_end)

    for node_idx in target_node_indices:
        
        if node_idx >= len(all_nodes):
            print(f"Warning: Node index {node_idx} out of bounds. Skipping.")
            continue

        current_node_id = all_nodes[node_idx]
        pos_curr = G.nodes()[current_node_id]["pos"]

        edges_at_node = list(G.edges(current_node_id))
        if not edges_at_node:
            print(f"Warning: Node {node_idx} has no connected edges. Skipping.")
            continue
            
        first_edge_at_node = edges_at_node[0]
        Area = G.edges[first_edge_at_node]["area"]

        outflow_non_rem = [sol[0](pos_curr) for sol in qps_non_rem]
        outflow_rem = [sol[0](pos_curr) for sol in qps_rem]

        velocity_non_rem = np.array(outflow_non_rem) / Area
        velocity_rem = np.array(outflow_rem) / Area


        v_nr_sliced = velocity_non_rem[idx_start_nr:idx_end_nr]
        t_nr_sliced = time_vec_non_rem[idx_start_nr:idx_end_nr]
        
        # Slice REM
        v_r_sliced = velocity_rem[idx_start_r:idx_end_r]
        t_r_sliced = time_vec_rem[idx_start_r:idx_end_r]

        # Calculate Means based on the SLICED window
        mean_val_non_rem = np.mean(v_nr_sliced) if len(v_nr_sliced) > 0 else 0
        mean_val_rem = np.mean(v_r_sliced) if len(v_r_sliced) > 0 else 0

        # Plotting
        fig, ax = plt.subplots(figsize=ps.FIG_SIZE)
        
        ax.plot(t_nr_sliced, v_nr_sliced, **ps.STYLE_NON_REM)
        ax.plot(t_r_sliced, v_r_sliced, **ps.STYLE_REM)
        
        ax.axhline(mean_val_non_rem, **ps.STYLE_MEAN_NON_REM)
        ax.axhline(mean_val_rem, **ps.STYLE_MEAN_REM)

        ax.set_title(f"Velocity at Node {node_idx}", fontsize=16)
        ax.set_xlabel("t' [sec]", fontsize=16)
        ax.set_ylabel("Velocity u' [mm/s]", fontsize=16)
        ax.grid(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True)
        fig.tight_layout() 
        
        # Unique filename using node index
        filename = f"velocity_at_node_{str(node_idx)}.png"
        
        file_out_rem = os.path.join(save_path_rem, filename) 
        fig.savefig(file_out_rem, dpi=300)
        
        file_out_non_rem = os.path.join(save_path_non_rem, filename)  
        fig.savefig(file_out_non_rem, dpi=300)

        plt.close(fig)
