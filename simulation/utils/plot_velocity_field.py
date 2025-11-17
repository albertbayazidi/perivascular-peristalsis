import os
import numpy as np
import matplotlib.pyplot as plt
import simulation.utils.common_plotting_style as ps

def save_velocity_at_first_node(G, qps_rem, qps_non_rem, exp_data_rem, exp_data_non_rem, exp_folder_rem, exp_folder_non_rem, id, plot_window=100):

    save_path_rem = os.path.join(exp_folder_rem, "plots","velocity")
    save_path_non_rem = os.path.join(exp_folder_non_rem, "plots","velocity")

    os.makedirs(save_path_rem, exist_ok=True)
    os.makedirs(save_path_non_rem, exist_ok=True)

    all_nodes = list(G.nodes())

    n0 = all_nodes[0]
    pos0 = G.nodes()[n0]["pos"]

    T_non_rem = exp_data_non_rem["T"]
    time_steps_non_rem = len(qps_non_rem)
    time_vec_non_rem = np.linspace(0, T_non_rem, time_steps_non_rem)

    T_rem = exp_data_rem["T"]
    time_steps_rem = len(qps_rem)
    time_vec_rem = np.linspace(0, T_rem, time_steps_rem)

    for e in G.edges():
        radius1 = G.edges()[e]["radius1"]
        radius2 = G.edges()[e]["radius2"]
        G.edges()[e]["area"] = np.pi * (radius2**2 - radius1**2)

    edges_at_n0 = list(G.edges(n0))

    first_edge_at_n0 = edges_at_n0[0]
    A0 = G.edges[first_edge_at_n0]["area"]

    outflow_at_n0_non_rem = [sol[0](pos0) for sol in qps_non_rem]
    velocity_at_n0_non_rem = np.array(outflow_at_n0_non_rem) / A0

    outflow_at_n0_rem = [sol[0](pos0) for sol in qps_rem]
    velocity_at_n0_rem = np.array(outflow_at_n0_rem) / A0

    mean_val_non_rem = np.mean(velocity_at_n0_non_rem[:plot_window])
    mean_val_rem = np.mean(velocity_at_n0_rem[:plot_window])

    fig, ax = plt.subplots(figsize=ps.FIG_SIZE)
    ax.plot(time_vec_non_rem[:plot_window], velocity_at_n0_non_rem[:plot_window], **ps.STYLE_NON_REM)
    ax.plot(time_vec_rem[:plot_window], velocity_at_n0_rem[:plot_window], **ps.STYLE_REM)
    ax.axhline(mean_val_non_rem, **ps.STYLE_MEAN_NON_REM)
    ax.axhline(mean_val_rem, **ps.STYLE_MEAN_REM)

    ax.set_title(f"Velocity at First Node", fontsize=16)
    ax.set_xlabel("t' [s]", fontsize=16)
    ax.set_ylabel("Velocity u' [mm/s]", fontsize=16)
    ax.grid(True)
    ax.legend()
    fig.tight_layout() 
    
    file_out_rem = os.path.join(save_path_rem,f"velocity_at_node{str(n0)}_{str(id)}.png") 
    fig.savefig(file_out_rem, dpi=300)
    
    file_out_non_rem = os.path.join(save_path_non_rem,f"velocity_at_node{str(n0)}_{str(id)}.png")  
    fig.savefig(file_out_non_rem, dpi=300)

    plt.close(fig)
