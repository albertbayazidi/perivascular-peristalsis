from graphnics import *
from xii import *
import networkx as nx

from graph.generate_simple_arterial_tree import make_arterial_tree
from graph.graph_utils import add_position_weights 

def setup_graph(depth, betas, Ls, radius0):
    """
    Make a graph with 1, 2 or n vessels, with lengths Ls and aspect ratios betas
    """

    if len(betas) == 1 and depth == 1:
        G = line_graph(n=2, dim=2, dx=Ls[0])
        G.nodes[0]["depth"] = depth
        
    elif len(betas) == 2 and depth == 1:
        La, Lb = Ls
        
        G = line_graph(n=3, dim=2, dx=La)

        G.nodes()[0]["pos"] = [0, 0]
        G.nodes()[1]["pos"] = [La, 0]
        G.nodes()[2]["pos"] = [La + Lb, 0]
        G.nodes[0]["depth"] = depth

    elif len(betas) == 3 and depth == 2:
        G = Y_bifurcation()

        La, Lb, Lc = Ls
        G.nodes()[0]["pos"] = [0, 0]
        G.nodes()[1]["pos"] = [0, La]
        G.nodes()[2]["pos"] = [-np.sqrt(0.5) * Lb, La + np.sqrt(0.5) * Lb]
        G.nodes()[3]["pos"] = [np.sqrt(0.5) * Lc, La + np.sqrt(0.5) * Lc]
        G.nodes[0]["depth"] = depth

    else:
        signs = np.tile([-1, 1], 10).tolist()
        signs[0] = 1
        signs[3] = 1
        signs[4] = 1

        G = make_arterial_tree(depth, directions=signs, gam=0.8, L0=Ls[0], radius0=radius0)

        for e in G.edges():
            radius0 = G.edges()[e]["radius"]
            G.edges()[e]["radius1"] = radius0
            G.edges()[e]["radius2"] = radius0 * betas[0]
            G.edges()[e]["beta"] = betas[0]

        G.make_mesh(1)
        G.compute_edge_lengths()
        G.nodes[0]["depth"] = depth

        add_position_weights(G)
        G.nodes[0]["longest_path"]  = nx.dag_longest_path_length(G, weight='weight')

        return G

    # Add inner and outer radiuses, and beta values, as edge attributes
    nx.set_edge_attributes(G, radius0, "radius1")

    # make dict of betas
    betas_dict = {e: beta for e, beta in zip(G.edges(), betas)}
    nx.set_edge_attributes(G, betas_dict, "beta")

    radius2_dict = {e: beta * radius0 for e, beta in zip(G.edges(), betas)}
    nx.set_edge_attributes(G, radius2_dict, "radius2")

    G.make_mesh(1)

    add_position_weights(G)
    G.nodes[0]["longest_path"]  = nx.dag_longest_path_length(G, weight='weight')
    return G
