import numpy as np
import numpy.linalg
import ufl

def _invR(beta):
    "Helper function for evaluating \mathcal{R}^{-1}(beta)."
    
    val = np.pi/8*(beta**4 - 1 - (beta**2 - 1)**2/ufl.ln(beta))
    return val

def _R(beta):
    "Helper function for evaluating \mathcal{R}(beta)."
    return 1.0/_invR(beta)

def _delta(beta):
    "Helper function for evaluating Delta(beta)."
    delt = ((2 - (beta**2-1)/ufl.ln(beta))**2)/(beta**4 - 1 - (beta**2 - 1)**2/ufl.ln(beta))
    return delt
    
def _beta(r_e, r_o):
    return r_e/r_o

def _alpha(l, P, R):
    "Helper function for matrix/vector expression."
    z = 1j
    A1 = (0.5 - (1 - np.cos(l))/(l**2))
    A2 = P*(1 - np.exp(z*l))/(2*l**2*R)
    return (A1 + A2.real)

def _xi(l):
    "Helper function for matrix/vector expression."
    z = 1j
    xi1 = (np.exp(z*l) - 1)/l
    return xi1



def get_Q_single(beta, l, epsilon):
    delta = _delta(beta)
    return epsilon*delta*(0.5 - (1 - np.cos(l))/(l**2))
    

def get_Q_tandem(r0, betas, ls, epsilon):
    '''
    Get expected flow rate in tandem vessels a and b
    
    Args:
        betas (list): aspect ratios of R1_a vs R2_a and R1_b vs R2_b
        ls (list): fractional length of vessel a and b
        epsilon (float): amplitude of vasomotion
        
    Returns: <Q> computed using (27)
    '''
    
    Delta_a, Delta_b = [_delta(beta) for beta in betas]
    Res_a, Res_b = [_R(beta)/r0**4 for beta in betas]
    la, lb = ls
    
    term1 = (Delta_a*Res_a*la + Delta_b*Res_b*lb)/(2*(Res_a*la+Res_b*lb))
    
    term2 = (Delta_a*Res_a**2*(1-np.cos(la)) + Delta_b*Res_b**2*(1-np.cos(lb)))/(Res_a*la + Res_b*lb)**2
    
    term3 = (Delta_a + Delta_b)*Res_a*Res_b/(2*(Res_a*la + Res_b*lb)**2)*(1-np.cos(la)-np.cos(lb)+np.cos(la+lb))
    
    return epsilon*(term1 - term2 + term3)



def solve_for_P(indices, paths, R, ell):

    # A bifurcating tree has N junctions with n = 2N + 1 edges. The
    # number of junctions N plus number of downstream ends (N+1) is
    # also 2N + 1

    # Convention: junction index == index of mother edge (in
    # bifurcating trees)

    # n x n system of linear equations for determining P: A P = b (complex):
    n = len(R)
    A = np.zeros((n, n), dtype=complex)
    b = np.zeros(n, dtype=complex)
    
    # The complex number i
    z = 1j

    for (i, j, k) in indices:
        # Convention: junction index == index of mother edge (in
        # bifurcating trees)
        I = i 

        # Set the correct matrix columns for this junction constraint
        A[I, i] = np.exp(z*ell[i])/(R[i]*ell[i])
        A[I, j] = - 1.0/(R[j]*ell[j])
        A[I, k] = - 1.0/(R[k]*ell[k])

        # Set the correct vector row for this junction constraint
        b[I] = _xi(ell[i]) + z - _xi(-ell[j]) - _xi(-ell[k])

    # Apply the addition degrees of freedom
    I = len(indices)
    for (k, path) in enumerate(paths):
        x_n = 0.0
        for n in path:
            A[I+k, n] = np.exp(-z*x_n)
            x_n += ell[n]

    # Solve the linear systems for real and imaginary parts of P:
    P = np.linalg.solve(A, b)
    
    return P

def solve_for_dP(R, ell, Delta, indices, paths, P):

    # A bifurcating tree has N junctions with 2N + 1 edges. The number
    # of junctions N plus number of downstream ends (N+1) is also 2N +
    # 1
    n = len(P)
    
    # n x n system of linear (real) equations for determining dP: A dP = b 
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    # Define the real linear system A P = b 
    for (i, j, k) in indices:
        # Convention: junction index == index of mother edge (in
        # bifurcating trees)
        I = i 
        
        # Set right matrix columns for this junction constraint
        A[I, i] = 1.0/(R[i]*ell[i])
        A[I, j] = - 1.0/(R[j]*ell[j])
        A[I, k] = - 1.0/(R[k]*ell[k])

        # Set right vector row for this junction constraint
        b[I] = (Delta[i]*_alpha(ell[i], P[i], R[i]) 
                - Delta[j]*_alpha(ell[j], P[j], R[j])
                - Delta[k]*_alpha(ell[k], P[k], R[k]))

    # Define the additional constraints:
    I = len(indices)
    for (k, path) in enumerate(paths):
        for n in path:
            A[I+k, n] = 1.0
            
    # Solve the linear systems for real and imaginary parts of P:
    dP = np.linalg.solve(A, b)
    
    return dP

def avg_Q_1_n(P, dP, ell, R, Delta):  
    "Evaluate <Q_1_n>."
    z = 1j
    val = (- dP/(R*ell) + Delta*(1./2 - (1 - np.cos(ell))/(ell**2)) 
           + Delta/(2*ell**2*R)*(P*(1 - np.exp(z*ell))).real)
    return val


def solve_bifurcating_tree(indices, paths, beta, ell):
    
    # Computation of dimension-less parameters
    R = [_R(b) for b in beta]
    Delta = [_delta(b) for b in beta]
    
    print("Solving for P")
    P = solve_for_P(indices, paths, R, ell)
    
    print("Solving for dP")
    dP = solve_for_dP(R, ell, Delta, indices, paths, P)

    print("Evaluating the flow rates Q1[i]: ...")
    Q1 = np.zeros(len(dP))
    for (i, _) in enumerate(dP):
        Q1[i] = avg_Q_1_n(P[i], dP[i], ell[i], R[i], Delta[i])

    return (P, dP, Q1)

def solve_three_junction(indices, paths, r_o, r_e, L, k):

    # Computation of dimension-less parameters
    beta = [_beta(re, ro) for (re, ro) in zip(r_e, r_o)]
    ell = [k*l for l in L]
    R = [_R(b) for b in beta]
    Delta = [_delta(b) for b in beta]
    print("beta = ", beta)
    print("ell = ", ell)
    print("R = ", R)
    print("Delta = ", Delta)
    
    n = 3
    A = np.zeros((n, n), dtype=complex)
    b = np.zeros(n, dtype=complex)
    
    # The complex number i
    z = 1j

    # Define the complex linear system A P = b in terms of the real
    # and imaginary part: Ar Pr = Br and Ai Pi = Bi:

    # Set right matrix columns for this junction constraint
    # [P0, P1, P3]
    A[0, :] = [np.exp(z*ell[0])/(R[0]*ell[0]), - 2*1./(R[1]*ell[1]), 0]
    A[1, :] = [0, np.exp(z*ell[1])/(R[1]*ell[1]), - 2*1./(R[3]*ell[3])]
    A[2, :] = [1, np.exp(-z*ell[0]), np.exp(-z*ell[0] - z*ell[1])]

    # Set right vector row for this junction constraint
    b[0] = _xi(ell[0]) + z - 2*_xi(-ell[1])  
    b[1] = _xi(ell[1]) + z - 2*_xi(-ell[3])  
    b[2] = 0

    
    # Solve the linear systems for real and imaginary parts of P:
    print("Solving A P = b")
    P = np.linalg.solve(A, b)
    (P0, P1, P3) = P

    P = (P0, P1, P1, P3, P3, P3, P3)

    print("Solving for dP")
    dP = solve_for_dP(R, ell, Delta, indices, paths, P)

    print("Evaluating the flow rates Q1[i]: ...")
    Q1 = np.zeros(len(dP))
    for (i, _) in enumerate(dP):
        Q1[i] = avg_Q_1_n(P[i], dP[i], ell[i], R[i], Delta[i])
    
    return (P, dP, Q1)



def single_bifurcation_data():

    # Data structure for representing network:
    # For each junction, list (mother edge no., daughter edge no, daughter edge no.)
    indices = [(0, 1, 2),]
    paths = [(0, 1), (0, 2)]
    
    # Peristaltic wave parameters: wave length lmbda and (angular) wave number k
    f = 1.0                 # frequency (Hz = 1/s)
    omega = 2*np.pi*f          # Angular frequency (Hz)
    lmbda = 1.0             # mm    
    k = 2*pi/lmbda          # wave number (1/mm)
    varepsilon = 0.1       # AU 
    ro0 = 0.1              # Base inner radius (mm)
    re0 = 0.2              # Base outer radius (mm)
    ro1 = ro0
    re1 = re0
        
    r_o = [ro0, ro1, ro1]  # Inner radii (mm) for each element/edge
    r_e = [re0, re1, re1]  # Outer radii (mm) for each element/edge
    L = [1.0, 1.0, 1.0]    # Element lengths (mm)
        
    #print("network = ", indices)
    #print("paths = ", paths)
    print("r_o = ", r_o)
    print("r_e = ", r_e)
    print("L = ", L)
    print("k = ", k)
    print("varepsilon = ", varepsilon)
    
    return (indices, paths, r_o, r_e, L, k, omega, varepsilon)

def three_junction_data():
    
    # Data structure for representing network:
    # For each junction, list (mother edge no., daughter edge no, daughter edge no.)
    indices = [(0, 1, 2), (1, 3, 4), (2, 5, 6)]
    paths = [(0, 1, 3), (0, 1, 4), (0, 2, 5), (0, 2, 6)]
    
    # Peristaltic wave parameters: wave length lmbda and (angular) wave number k
    f = 1.0                 # frequency (Hz = 1/s)
    omega = 2*np.pi*f          # Angular frequency (Hz)
    lmbda = 2.0             # mm    
    k = 2*np.pi/lmbda          # wave number (1/mm)
    varepsilon = 0.1        # AU 
    ro0 = 0.01              # Base inner radius (mm)
    re0 = 0.02              # Base outer radius (mm)
    ro1 = ro0/2; ro2 = ro1/2
    re1 = re0/2; re2 = re1/2
        
    r_o = [ro0, ro1, ro1, ro2, ro2, ro2, ro2]  # Inner radii (mm) for each element/edge
    r_e = [re0, re1, re1, re2, re2, re2, re2]  # Outer radii (mm) for each element/edge
    L = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]    # Element lengths (mm)
        
    #print("network = ", indices)
    #print("paths = ", paths)
    print("r_o = ", r_o)
    print("r_e = ", r_e)
    print("L = ", L)
    print("k = ", k)
    print("varepsilon = ", varepsilon)

    return (indices, paths, r_o, r_e, L, k, omega, varepsilon)

def generate_murray_tree(m, r=0.1, gamma=1.0, beta=2.0, L0=10):
    """Generate a Murray tree of m (int) generations. 

    Starting one branch for m = 0. Given r is the radius of the mother
    branch. Radii of daughter branches (j, k) with parent i is
    goverend by gamma, such that

    r_o_j + r_o_k = r_o_i.

    and

    r_o_j/r_o_k = gamma

    i.e 

    r_o_j = gamma r_o_k

    """
    
    # Compute the number of elements/branches
    N = int(sum([2**i for i in range(m+1)]))

    # Create the generations
    generations = [[0,]]
    for i in range(1, m+1):
        _N = int(sum([2**j for j in range(i)]))
        siblings = list(range(_N, _N + 2**i))
        generations.append(siblings)

    # Create the indices (bifurcations)
    indices = []
    for (i, generation) in enumerate(generations[:-1]):
        children = generations[i+1]
        for (j, parent) in enumerate(generation):
            (sister, brother) = (children[2*j], children[2*j+1])
            indices += [(parent, sister, brother)]
    n = len(indices)
    assert N == (2*n+1), "N = 2*n+1"

    # Iterate through generations from children and upwards by
    # reversing the generations list for creating the paths
    generations.reverse()

    # Begin creating the paths
    ends = generations[0]
    paths = [[i,] for i in ends]
    for (i, generation) in enumerate(generations[1:]):
        for (j, path) in enumerate(paths):
            end_node = path[0]
            index = j // (2**(i+1))
            path.insert(0, generations[i+1][index])

    # Reverse it back
    generations.reverse()

    # Create inner radii based on Murray's law with factor gamma
    r_o = numpy.zeros(N)
    r_o[0] = r  
    for (g, generation) in enumerate(generations[1:]):
        for index in generation[::2]:
            ri = r_o[0]/(2**g)
            rk = 1/(1+gamma)*ri
            rj = gamma*rk
            r_o[index] = rk
            r_o[index+1] = rj
            
    L = L0*r_o
    r_e = beta*r_o
        
    return (generations, indices, paths, r_o, r_e, L)
    

def Qprime(Q, varepsilon, omega, L, k, Delta, r_o):
    scale = 2*np.pi*varepsilon*omega*r_o**2/k
    val = scale*Q
    return val

if __name__ == "__main__":

    import sys
    
    if False:
        print("Solving Murray tree.")
        (generations, indices, paths, r_o, r_e, L) = \
            generate_murray_tree(int(sys.argv[1]), r=0.1, gamma=0.8, beta=2.0, L0=10)
        # Peristaltic wave parameters: wave length lmbda and (angular) wave number k
        f = 1.0                 # frequency (Hz = 1/s)
        omega = 2*pi*f          # Angular frequency (Hz)
        lmbda = 1.0             # mm    
        k = 2*pi/lmbda          # wave number (1/mm)
        varepsilon = 0.1        # AU 
        #print(indices)
        #print(paths)
        print(r_o)
        print(r_e)
        print(L)
        (P, dP, avg_Q_1) = solve_bifurcating_tree(indices, paths, r_o, r_e, L, k)
        
    if False:
        (indices, paths, r_o, r_e, L, k, omega, varepsilon) = single_bifurcation_data()
        print("Solving general tree (single bifurcation case)")
        (P, dP, avg_Q_1) = solve_bifurcating_tree(indices, paths, r_o, r_e, L, k)

    if False:
        (indices, paths, r_o, r_e, L, k, omega, varepsilon) = three_junction_data()
        print("Solving three-junction case explicitly")
        (P, dP, avg_Q_1) = solve_three_junction(indices, paths, r_o, r_e, L, k)

    if False:
        (indices, paths, r_o, r_e, L, k, omega, varepsilon) = three_junction_data()
        print("Solving general tree (three junction case)")
        (P, dP, avg_Q_1) = solve_bifurcating_tree(indices, paths, r_o, r_e, L, k)

    if False:
        print("P = ", P)
        print("dP = ", dP)
        print("<Q_1> = ", avg_Q_1)
        print("eps*<Q_1_0> = %.3e" % (varepsilon*avg_Q_1[0]))
        
        beta0 = _beta(r_e[0], r_o[0])
        delta0 = _delta(beta0)
        Q10 = varepsilon*avg_Q_1[0]
        Qp = Qprime(Q10, varepsilon, omega, L[0], k, delta0, r_o[0])
        print("eps*<Q_1_0>' (mm^3/s) = %.3e" % Qp)
            
