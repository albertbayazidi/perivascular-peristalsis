import numpy as np
import networkx as nx
from analytics.new_implementation.helper_functions import _delta,_R
from analytics.new_implementation.pressure_system import construct_dP_matrix,construct_P_matrix,solve_system
from analytics.new_implementation.utils.graph_utils import junctions, find_twig_path

def avg_Q_1_n(P, dP, ell, R, Delta):  
    "Evaluate <Q_1_n>."
    z = 1j
    val = (- dP/(R*ell) + Delta*(1./2 - (1 - np.cos(ell))/(ell**2)) 
           + Delta/(2*ell**2*R)*(P*(1 - np.exp(z*ell))).real)
    return val


def get_Q(r0, betas, ls, eps, G):
    nr_of_junctions = np.log2( len(G.nodes)  ) - 1
   
    if nr_of_junctions == 1:
        # DO modifications needed for single junction comparison
        gamma = np.full(len(betas),1)
    else:
        gamma = np.asarray(list(nx.get_edge_attributes(G, 'radius').values()))/r0

    indices = junctions(G)
    paths = find_twig_path(G)

    R = [_R(b) for b in betas]
    Delta = [_delta(b) for b in betas]
    
    A,b = construct_P_matrix(indices, paths, ls, R, gamma)
    P,_ = solve_system(A,b)

    A,b = construct_dP_matrix(R, ls, Delta, indices, paths, P, gamma)
    dP,_ = solve_system(A,b)
    
    Q1s = np.zeros(len(dP))

    for (i, _) in enumerate(dP):
        Q1s[i] = avg_Q_1_n(P[i], dP[i], ls[i], R[i], Delta[i]).real # to suppress warnings

    Q = [Q1*eps for Q1 in Q1s]
     
    return Q

