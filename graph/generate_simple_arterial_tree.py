from graph.spatial_utils import compute_vessel_endpoint, Point, doIntersect
from graphnics import copy_from_nx_graph
import copy

from graph.graph_utils import junctions, find_twig_path

import networkx as nx
import numpy as np
import random

random.seed(42)


def _try_add_bifurcation(G, parent_node, new_node_ix, gam, lmbda, normal, sign_choice, uniform_lengths, L0):
    """
    Attempts to add a new bifurcation at a given parent_node.
    Returns (success, updated_node_index, list_of_new_edges).
    """
    try:
        parent_edge = list(G.in_edges(parent_node))[0]
    except IndexError:
        return False, new_node_ix, [] # Cannot bifurcate from the root node

    # Parent vessel properties
    previousvessel = [G.nodes[parent_edge[0]]["pos"], G.nodes[parent_node]["pos"]]
    D0 = G.edges[parent_edge]["radius"] * 2

    # Daughter diameters and lengths
    D2 = D0 * (gam**3 + 1) ** (-1 / 3)
    D1 = gam * D2
    L2 = lmbda * D2
    L1 = lmbda * 0.6 * D1

    if uniform_lengths:
        L1, L2 = L0, L0

    # Bifurcation angles
    cos1 = (D0**4 + D1**4 - (D0**3 - D1**3) ** (4 / 3)) / (2 * D0**2 * D1**2)
    angle1 = np.degrees(np.arccos(cos1))
    cos2 = (D0**4 + D2**4 - (D0**3 - D2**3) ** (4 / 3)) / (2 * D0**2 * D2**2)
    angle2 = np.degrees(np.arccos(cos2))

    sign1 = sign_choice
    sign2 = -1 * sign1

    branch1 = [sign1, angle1, L1, D1]
    branch2 = [sign2, angle2, L2, D2]

    potential_branches = []
    all_pos = nx.get_node_attributes(G, "pos")
    non_neighbor_edges = [edge for edge in G.edges() if parent_node not in edge]

    # Check that both new branches do not cause collisions
    for sign, angle, L, D in [branch1, branch2]:
        new_node_pos = compute_vessel_endpoint(
            previousvessel, normal(*previousvessel[1]), sign * angle, L
        )

        C = Point(G.nodes[parent_node]["pos"])
        DD = Point(new_node_pos)

        # Check for intersection with existing non-neighbor edges
        for v1, v2 in non_neighbor_edges:
            A = Point(all_pos[v1])
            B = Point(all_pos[v2])
            if doIntersect(A, B, C, DD):
                return False, new_node_ix, [] # Collision detected, abort this bifurcation

        potential_branches.append({"pos": new_node_pos, "radius": D / 2})

    # If no collisions were found for either branch, add them to the graph
    new_edges = []
    for branch_data in potential_branches:
        new_node_ix += 1
        new_edge = (parent_node, new_node_ix)
        G.add_edge(*new_edge)
        G.nodes[new_node_ix]["pos"] = branch_data["pos"]
        G.edges[new_edge]["radius"] = branch_data["radius"]
        new_edges.append(new_edge)

    return True, new_node_ix, new_edges


def make_arterial_tree(N, radius0=1, gam=0.8, L0=3, directions=False, uniform_lengths=False):
    """
    N (int): number of levels in the arterial tree
    radius0 (float): radius of first vessel
    gam (float): ratio between daughter vessel radii
    directions (list): vector of choices (+-1) of vessel direction. If no vector is given this is assigned randomly.
    uniform_lengths (bool): uniform branch length

    Uniform lengths is typically only of interest in numerical tests
    Assigning directions is useful for reproducability of results.
    """
    # Parameters
    # Origin location
    p0 = [0, 0, 0]
    initial_direction = [0, 1, 0]
    D0 = 2 * radius0
    lmbda = L0 / D0

    # By convention we chose gam <=1 so D1 will always be smaller or equal to D2
    if gam > 1:
        raise Exception("Please choose a value for gamma lower or equal to 1")

    # Surface normal function
    # The surface normal here is fixed because we want to stay in the x,y plane.
    # But this could be the normal of any surface.
    def normal(x, y, z):
        return [0, 0, 1]

    # Set up direction choices
    if directions:
        directions_iter = iter(directions)
        get_sign = lambda: next(directions_iter, random.choice((-1, 1)))
    else:
        get_sign = lambda: random.choice((-1, 1))

    #### Creation of the graph
    G = nx.DiGraph()
    G.add_edge(0, 1)
    nx.set_node_attributes(G, {0: p0}, "pos")
    nx.set_edge_attributes(G, D0 / 2, "radius")
    G.nodes[1]["pos"] = np.asarray(p0) + np.asarray(initial_direction) * L0
    new_node_ix = 1

    #### Phase 1: Iteratively build the tree for N generations
    previous_edges = [(0, 1)]
    for igen in range(1, N):
        current_edges = []
        for u, v in previous_edges:
            parent_node = v
            success, new_node_ix, new_edges = _try_add_bifurcation(
                G, parent_node, new_node_ix, gam, lmbda, normal, get_sign(), uniform_lengths, L0
            )
            if success:
                current_edges.extend(new_edges)
        previous_edges = current_edges
        if not previous_edges:
            print(f"Warning: Tree growth stopped at generation {igen} due to collisions.")
            break
            
    #### Phase 2: Add missing junctions until the tree is complete
    expected_junctions = 2 ** (N - 1) - 1
    while len(junctions(G)) < expected_junctions:
        possible_parents = [n for n in G.nodes() if G.out_degree(n) == 0 and G.in_degree(n) > 0]
        if not possible_parents:
            print("Warning: Could not add all missing junctions. No valid bifurcation points remain.")
            break

        bifurcation_added_in_pass = False
        for parent_node in possible_parents:
            success, new_node_ix, _ = _try_add_bifurcation(
                G, parent_node, new_node_ix, gam, lmbda, normal, get_sign(), uniform_lengths, L0
            )
            if success:
                bifurcation_added_in_pass = True
                break # Restart while-loop with the updated graph

        if not bifurcation_added_in_pass:
            print("Warning: Could not find a valid location for any new junction.")
            break

    # Convert to FenicsGraph
    G_ = nx.convert_node_labels_to_integers(G)
    G = copy_from_nx_graph(G_)
    G.make_mesh(3)
    
    return G
