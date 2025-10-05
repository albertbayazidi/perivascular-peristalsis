from graph.spatial_utils import compute_vessel_endpoint, Point, doIntersect
from graphnics import copy_from_nx_graph
import copy

import networkx as nx
import numpy as np
import random

random.seed(42)



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
    L0 = L0 +3
    # Initial direction
    direction = [0, 1, 0]

    # First vessel diameter
    D0 = 2 * radius0

    # By convention we chose gam <=1 so D1 will always be smaller or equal to D2
    if gam > 1:
        raise Exception("Please choose a value for gamma lower or equal to 1")

    # Surface normal function
    # The surface normal here is fixed because we want to stay in the x,y plane.
    # But this could be the normal of any surface.
    def normal(x, y, z):
        return [0, 0, 1]

    #### Creation of the graph

    # Create a networkx graph
    G = nx.DiGraph()

    # Create the first vessel
    #########################
    L = L0
    lmbda = L / (D0)

    G.add_edge(0, 1)
    nx.set_node_attributes(G, p0, "pos")
    nx.set_edge_attributes(G, D0 / 2, "radius")

    G.nodes[1]["pos"] = np.asarray(p0) + np.asarray(direction) * L
    new_node_ix = 1

    #### Iteration to create the other vessels following a given law

    # list of the vessels from the previous generation
    previous_edges = [(0, 1)]

    for igen in range(1, N):
        current_edges = []
        for e in previous_edges:
            # Parent vessel properties
            previousvessel = [G.nodes[e[0]]["pos"], G.nodes[e[1]]["pos"]]
            D0 = G.edges[e]["radius"] * 2

            # Daughter diameters
            D2 = D0 * (gam**3 + 1)**(-1 / 3)
            D1 = gam * D2
            # Daughter lengths
            L2 = lmbda * D2
            L1 = lmbda * 0.6 * D1

            if uniform_lengths:
                L1, L2 = L, L
            # Bifurcation angles
            # angle for the smallest vessel
            cos1 = (D0**4 + D1**4 -
                    (D0**3 - D1**3)**(4 / 3)) / (2 * D0**2 * D1**2)
            angle1 = np.degrees(np.arccos(cos1))
            # angle for the biggest vessel
            cos2 = (D0**4 + D2**4 -
                    (D0**3 - D2**3)**(4 / 3)) / (2 * D0**2 * D2**2)
            angle2 = np.degrees(np.arccos(cos2))

            # direction-vector choose which vessel go to the right/left

            if not directions:
                sign1 = random.choice((-1, 1))
            else:
                sign1 = directions[0]
                del directions[0]
            sign2 = -1 * sign1

            branch1 = [sign1, angle1, L1, D1]
            branch2 = [sign2, angle2, L2, D2]

            parent_edge_v2 = e[1]  # vertex we want to connect to

            # Check that the new edge does not overlap other edges
            for sign, angle, L, D in [branch1, branch2]:
                new_node_pos = compute_vessel_endpoint(
                    previousvessel, normal(*previousvessel[1]), sign * angle,
                    L)

                all_pos = np.asarray(
                    list(nx.get_node_attributes(G, "pos").values()))
                intersecting_lines = 0
                C = Point(all_pos[parent_edge_v2])
                DD = Point(new_node_pos)

                non_neighbor_edges = copy.deepcopy(list(G.edges()))

                neighbor_edges = copy.deepcopy(
                    list(G.in_edges(parent_edge_v2)) +
                    list(G.out_edges(parent_edge_v2)))

                for en in list(set(neighbor_edges)):
                    non_neighbor_edges.remove(en)

                for v1, v2 in non_neighbor_edges:
                    A = Point(all_pos[v1])
                    B = Point(all_pos[v2])
                    if doIntersect(A, B, C, DD):
                        intersecting_lines += 1

                # if no edges overlap with this new one
                # we go ahead and add it
                if intersecting_lines < 1:
                    new_node_ix += 1

                    new_edge = (e[1], new_node_ix)
                    G.add_edge(*new_edge)

                    # Set the location according to length and angle
                    G.nodes[new_node_ix]["pos"] = new_node_pos

                    # Set radius
                    G.edges[new_edge]["radius"] = D / 2

                    # Add to the pool of vessels for this generation
                    current_edges.append(new_edge)

        previous_edges = current_edges

    # Convert to FenicsGraph
    G_ = nx.convert_node_labels_to_integers(G)

    G = copy_from_nx_graph(G_)
    G.make_mesh(3)

    return G
