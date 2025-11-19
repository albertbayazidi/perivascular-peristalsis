import networkx as nx
import numpy as np

def find_twig_path(G):
    leaves = [n for n in G.nodes if G.degree[n] == 1 and n != 0]
    paths = []
    for leaf in leaves:
        path = nx.shortest_path(G, source=0, target=leaf)
        path = path[1:] # ignore node 0
        path = np.array(path) - 1 # shift numbering down by 1
        paths.append(path.tolist())
    return paths

def junctions(G):
    junctions = [n for n in G.nodes if G.degree[n] == 3]

    junction_points = []
    for j in junctions:
        group = [j] + list(G.neighbors(j))
        group = np.array(group) - 1 # shift numbering down by 1 
        junction_points.append(group.tolist())

    return junction_points

def add_position_weights(G):
    for u, v in G.edges():
        p1 = np.array(G.nodes[u]["pos"])
        p2 = np.array(G.nodes[v]["pos"])
        w = np.linalg.norm(p1 - p2)
        G.edges[u, v]["weight"] = w

