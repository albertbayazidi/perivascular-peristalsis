import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_tree(G):
    pos = nx.get_node_attributes(G, 'pos')
    pos2d = [coord[0:2] for coord in list(pos.values())]

    radius1 = np.asarray(list(nx.get_edge_attributes(G, 'radius1').values()))
    radius2 = np.asarray(list(nx.get_edge_attributes(G, 'radius2').values()))

    fig, ax = plt.subplots(figsize=(5,5))

    nx.draw_networkx(G, pos2d, width=radius1*50, edge_color='firebrick', 
                    with_labels=False, node_size=0.05, node_color='firebrick', arrowsize=0.1, ax=ax)
    
    nx.draw_networkx(G, pos2d, width=radius2*60, edge_color='gray', 
                    with_labels=False, node_size=0.001, node_color='gray', arrowsize=0.1, ax=ax, alpha=0.5)

    
    ax.set_xlabel('x [mm]', fontsize=16)
    ax.set_ylabel('y [mm]', fontsize=16)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=12)
    fig.tight_layout() 
    plt.show()

    return fig, ax

