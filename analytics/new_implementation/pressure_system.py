import numpy as np
import scipy.sparse as sp
from analytics.new_implementation.helper_functions import _alpha, _xi
from analytics.new_implementation.utils.visualization_utils import *
from analytics.new_implementation.volemetric_flow import *

def solve_system(A, b):
    
    ilu = sp.linalg.spilu(A.T @ A)  
    Mx = lambda x: ilu.solve(x)
    M = sp.linalg.LinearOperator(A.shape, Mx)

    sol, info = sp.linalg.gmres(A.T @ A, A.T @ b, M=M, restart=200, maxiter=1000)
    return sol, info

def solve_system_lsqr(A, b):
    sol = sp.linalg.lsqr(A, b)
    return sol[0],sol[1:]

def solve_system_directly(A, b):
    sol = sp.linalg.spsolve(A, b)
    return sol

def construct_P_matrix(indices, paths, l, R, gamma):
    """
    Construct the system matrix A and RHS b.

    indices: list of (i,j,k) tuples for junctions
    paths: list of list, each path is a sequence of edge indices to a downstream end
    l, R, gamma: dicts keyed by edge index
    """
    z = 1j

    E = len(indices) + len(paths)
    A = sp.lil_matrix((E, E), dtype=complex) 
    b = np.zeros(E, dtype=complex) 
    
    for row,(i,j,k) in enumerate(indices):
        
        # (Eq. 38)
        A[row, i] = np.exp(1j*l[i])*gamma[i]/(R[i]* l[i])
        A[row, j] = - gamma[j]/(R[j]* l[j])
        A[row, k] = - gamma[k]/(R[k]* l[k])

        b[row] = ( gamma[i]* _xi(l[i]) - gamma[j]*_xi(-l[j]) - gamma[k]*_xi(-l[k])
                + z*(gamma[j] + gamma[k] - gamma[i]))

    # (Eq. 40)
    I = len(indices)
    for (k, path) in enumerate(paths):
        x_n = 0.0
        n = 0
        for n in path:
            A[I+k, n] = np.exp(-z*x_n)/gamma[n]
            x_n += l[n]

    return A.tocsr(),b


def construct_dP_matrix(R, l, Delta, indices, paths, P,gamma):

    # A bifurcating tree has N junctions with 2N + 1 edges. The number
    # of junctions N plus number of downstream ends (N+1) is also 2N +
    # 1
    E = len(P)
    
    # n x n system of linear (real) equations for determining dP: A dP = b 
    A = sp.lil_matrix((E, E), dtype=complex) 
    b = np.zeros(E, dtype=complex) 
    
    # Define the real linear system A P = b 
    for row,(i, j, k) in enumerate(indices):
        # Convention: junction index == index of mother edge (in
        # bifurcating trees)

        # Set right matrix columns for this junction constraint
        A[row, i] = 1.0/(R[i]* l[i])
        A[row, j] = - 1.0/(R[j]*l[j])
        A[row, k] = - 1.0/(R[k]*l[k])

        # Set right vector row for this junction constraint
        b[row] = (gamma[i] * Delta[i]*_alpha(l[i], P[i], R[i]) 
                - gamma[j] * Delta[j]*_alpha(l[j], P[j], R[j])
                - gamma[k] * Delta[k]*_alpha(l[k], P[k], R[k]))

    # Define the additional constraints:
    I = len(indices)
    for (k, path) in enumerate(paths):
        for n in path:
            A[I+k, n] = gamma[n]


    return A.tocsr(),b

    

