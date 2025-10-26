import os
import numpy as np
import matplotlib.pyplot as plt
from analytics.new_implementation.helper_functions import dimensional_Q

def save_velocity_at_first_node(G, experiments, exp_folder):
    save_path = os.path.join(exp_folder,"velocity.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    all_nodes = list(G.nodes())

    n0 = all_nodes[0]
    pos0 = G.nodes()[n0]['pos']
    
    for _, exp in enumerate(experiments):
        qps = exp["sol"]
        T = exp["T"]
        time_steps = len(qps)
        time_vec = np.linspace(0,T,time_steps)

        for e in G.edges():
            radius1 = G.edges()[e]['radius1']
            radius2 = G.edges()[e]['radius2']
            G.edges()[e]['area'] = np.pi * (radius2**2 - radius1**2)
            
        edges_at_n0 = list(G.edges(n0))
            
        first_edge_at_n0 = edges_at_n0[0]
        A0 = G.edges[first_edge_at_n0]['area']
        outflow_at_n0_over_time = [sol[0](pos0) for sol in qps]
        
        u, v = list(G.edges())[0]
        R0 = G.edges[u, v]["radius1"]

        Q_dim_at_n0_over_time = [
            float(dimensional_Q(Q, exp["k"], exp["w"], exp["epsilon"], R0))
            for Q in outflow_at_n0_over_time
        ]

        velocity_at_n0_over_time = np.array(Q_dim_at_n0_over_time) / A0
        
        plt.figure(figsize=(7, 5))
        plt.plot(time_vec, velocity_at_n0_over_time)
        plt.title(f"Velocity at First Node", fontsize=16)
        plt.xlabel("t' [s]", fontsize=16)
        plt.ylabel("Velocity u' [mm/s]", fontsize=16)
        plt.grid(True)
        
        file_out = os.path.splitext(save_path)[0] + f"_node{n0}.png"
        plt.tight_layout()
        plt.savefig(file_out, dpi=300)
        plt.close()
        
