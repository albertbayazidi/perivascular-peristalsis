import numpy as np
import os

from graphnics import *
from xii import *
import matplotlib.pyplot as plt

from analytics.new_implementation.helper_functions import dimensional_Q

def save_net_flow_at_first_node(G, experiments, exp_folder):
    save_path = os.path.join(exp_folder,"net_flow_at_")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_nodes = list(G.nodes())

    n0 = all_nodes[0]
    pos0 = G.nodes()[n0]['pos']

    for _, exp in enumerate(experiments):
        qps = exp["sol"]

        T = exp["T"]
        time_steps = len(qps)
        time_vec = np.linspace(0,T,time_steps)
        
        outflow_at_n0_over_time = [sol[0](pos0) for sol in qps]
        
        u, v = list(G.edges())[0]
        R0 = G.edges[u, v]["radius1"]
       
        Q_dim_at_n0_over_time = [
            float(dimensional_Q(Q, exp["k"], exp["w"], exp["epsilon"], R0))
            for Q in outflow_at_n0_over_time
        ]

        dt = T/time_steps
        ys = np.cumsum(Q_dim_at_n0_over_time) * dt
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        ax.plot(time_vec, ys)
        ax.set_title(f"Net flow at at First Node", fontsize=16)
        ax.set_xlabel("t' [s]", fontsize=16)
        ax.set_ylabel("$\\int_0^{t'} Q'(\\tau) \, \\mathrm{d} \\tau$ [$mm$]", fontsize=16)
        ax.grid(True)

        file_out = os.path.splitext(save_path)[0] + f"_node{n0}.png"
        fig.tight_layout()
        fig.savefig(file_out, dpi=300)
        plt.close(fig)

