import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def add_circles(G, node_ids, ax, radius=0.075):
    if not node_ids:
        return

    color_palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", 
                     "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080"]

    for i, node_id in enumerate(node_ids):
        current_color = color_palette[i % len(color_palette)]
        
        if node_id in G.nodes:
            node_pos = G.nodes[node_id]["pos"]
            
            circle = plt.Circle((node_pos[0], node_pos[1]), radius, 
                                color=current_color, alpha=0.7, zorder=3,
                                label=f"Node {node_id}") 
            ax.add_patch(circle)
        else:
            print(f"Warning: Node ID {node_id} not found in graph.")

def plot_tree(G, highlight_nodes=None):
    
    fig, ax = plt.subplots(figsize=(5, 7))
    pos = nx.get_node_attributes(G, "pos")

    if G.nodes[0].get("depth") == 1:
        pos2d = [(-coord[1], coord[0]) for coord in list(pos.values())]
        ax.set_xlabel("y [mm]", fontsize=16)
        ax.set_ylabel("x [mm]", fontsize=16)
    else:
        pos2d = [coord[0:2] for coord in list(pos.values())]
        ax.set_xlabel("x [mm]", fontsize=16)
        ax.set_ylabel("y [mm]", fontsize=16)
        
    if highlight_nodes:
        add_circles(G, highlight_nodes, ax)

    radius1 = np.asarray(list(nx.get_edge_attributes(G, "radius1").values()))
    radius2 = np.asarray(list(nx.get_edge_attributes(G, "radius2").values()))

    nx.draw_networkx(G, pos=pos2d, width=radius1 * 50, edge_color="firebrick",
                     with_labels=False, node_size=0.05, node_color="firebrick", 
                     arrowsize=0.1, ax=ax, style="solid")
    
    nx.draw_networkx(G, pos=pos2d, width=radius2 * 60, edge_color="gray",
                     with_labels=False, node_size=0.001, node_color="gray", 
                     arrowsize=0.1, ax=ax, alpha=0.5, style="solid")

    ax.legend(handles=[p for p in ax.patches if isinstance(p, plt.Circle)],
              loc="upper center", bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=12)
    ax.set_aspect("equal", adjustable="box") 
    fig.tight_layout()
    #ax.axis("off")
    plt.show()

    return fig, ax
