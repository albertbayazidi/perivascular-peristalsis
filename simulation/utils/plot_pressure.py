import os
import numpy as np
import matplotlib.pyplot as plt

from graphnics import *
from xii import *

from analytics.new_implementation.helper_functions import dimensional_P


import matplotlib.pyplot as plt
import numpy as np
import os

def save_pressure_field(G, experiments, exp_folder):
    save_path = os.path.join(exp_folder,"pressure.png")

    for _, exp in enumerate(experiments):  

        sol_list = exp["sol"]

        node_positions = np.array([G.nodes()[n]['pos'] for n in G.nodes()])
        x = node_positions[:, 0]
        y = node_positions[:, 1]

        pressure_per_node = [[sol[1](pos) for sol in sol_list] for pos in node_positions]

        pressures_at_T = [p_list[-1] for p_list in pressure_per_node]

        plt.figure(figsize=(7, 5))

        if np.ptp(y) > 1e-8:  # if y varies, plot in 2D
            plt.scatter(x, y, c=pressures_at_T, cmap="viridis", s=80)
            plt.colorbar(label="Pressure p(x, y, T)")
            plt.xlabel("x-position", fontsize=16)
            plt.ylabel("y-position", fontsize=16)
            plt.title(f"Pressure Field at Final Time T (2D)", fontsize=16)
        else:  # 1D line plot
            sort_idx = np.argsort(x)
            plt.plot(x[sort_idx], np.array(pressures_at_T)[sort_idx], marker="o")
            plt.xlabel("Position along 1D segment (x)", fontsize=16)
            plt.ylabel("Pressure p(x, T) dimensionless", fontsize=16)
            plt.title(f"Pressure Field at Final Time T (1D)", fontsize=16)
            plt.grid(True)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file_out = os.path.splitext(save_path)[0] + f".png"
        plt.tight_layout()
        plt.savefig(file_out, dpi=300)
        plt.close()

