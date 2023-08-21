from graphnics import *
from xii import *
import ufl
import numpy as np


import sys
sys.path.append('../analytics')
import pvs_network_netflow as pvs
from utils import *

'''
Helper functions that run a peristaltis simulation on a given arterial tree
'''

    
def run_peristalsis_simulation(G, lamdas, freqs, n_cycles, tsteps_per_cycle, epsilon=0.1):
    '''
    Simulate pulsatile flow due to vasomotion in an arterial tree
    
    Args:
        G: arterial tree
        lamdas (list): list of wave lengths
        freq (float): frequency of vasomotion
        beta (float): aspect ratio of R1 vs R2 (so R2 = beta*R1)
        n_cycles (int): number of cycles to simulate
        tsteps_per_cycle (int): number of time steps per cycle
        epsilon (float): amplitude of vasomotion
        
    Returns:
        experiments (list): list of dicts containing parameters and results
    '''
    
    G.compute_edge_lengths()
    lengths = [G.edges()[e]['length'] for e in G.edges()]
    total_length = sum(lengths)
    
    # We make a list of "experiments" containing dicts that stores parameters and results
    experiments = []
    for lamda in lamdas:
        for freq in freqs:
            for i, tstep_per_cycle in enumerate(tsteps_per_cycle):
                
                params = {'k':get_k(lamda), 
                          'w':get_w(freq), 
                          'epsilon':epsilon, 
                          'freq':freq, 
                          'lamda':lamda, 
                          'l':get_k(lamda)*total_length, 
                          'ts_per_cycle':tstep_per_cycle,
                           'n_cycles':n_cycles[i], 
                           'T':n_cycles[i]/freq}
                
                experiments.append(params)
    
    
    for exp in experiments:
        print('Solving...')
        
        # adjust mesh size to <1/200 of the wave length
        i = 3
        G.make_mesh(i)
        while G.mesh.hmax() > exp['lamda']/100:        
            G.make_mesh(i)
            i += 1
            
        lamda = exp['lamda']
        print(f'lambda={lamda}, number of mesh refinements: {i}')
        
        # Vasomotion model is expressed via ufl functions depending
        # on constant t_, k_, w_ and s_
        f, area_inv, res, g, t_, k_, w_, s_, eps_, R1 = peristalsis_as_ufl(G)

        
        for e in G.edges():
            G.edges()[e]['Res'] = res
            G.edges()[e]['Ainv'] = area_inv
        
        
        t_.assign(0)
        k_.assign(exp['k'])
        w_.assign(exp['w'])
        eps_.assign(epsilon)
        
        n_cycles = exp['n_cycles']
    
        model = TimeDepHydraulicNetwork(G, p_bc=Constant(0), f=f, Ainv=area_inv, Res=res, degree=1)
        
        time_steps = exp['ts_per_cycle']*n_cycles
        
        qps = time_stepping_stokes(model, t=t_, t_steps = time_steps, T=exp['T'], t_step_scheme='IE')
        
        exp['sol'] = qps

        
        # also store the graph
        exp['G'] = G
        
        
    return experiments

def peristalsis_as_ufl(G):
    """
    Get a ufl expression for the inner radius
        R_0 = R_0(1+epsilon*sin(k*s-w*t))
    of a vessel undergoing vasomotion.
    
    Args:
        G (nx.Graph): graph representing the network
    The edges of the graph must have the following attributes:
        'radius1' (float): radius of the inner boundary at rest
        'radius2' (float): radius of the outer boundary
        'Res' (float): resistance
                
    Returns:
        f (ufl.Expression): source term
        Ainv (ufl.Expression): inverse of cross sectional area
        Res (ufl.Expression): resistance
        g (ufl.Expression): force term
        t_ (ufl.Constant): time
        k_  (ufl.Constant): wave number
        w_ (ufl.Constant): wave frequency
        s_ (ufl.Expression): distance from source
        eps_ (ufl.Constant): amplitude of vasomotion
        R1 (ufl.Expression): radius of the vessel
        
        
    """
    
    P1 = FunctionSpace(G.mesh, 'CG', 1)
    
    # Fenics constants that parametrize the vasomotion model
    dist = DistFromSource(G, 0)
    s_ = interpolate(dist, P1) # distance from source, computeded using bfs
    t_ = Constant(0)
    k_ = Constant(1)
    w_ = Constant(1)
    eps_ = Constant(0.1)
    
    
    R1_0 = nxgraph_attribute_to_dolfin(G, 'radius1') # inner radius at rest
    R2 = nxgraph_attribute_to_dolfin(G, 'radius2') # outer radius, constant in time
    beta = nxgraph_attribute_to_dolfin(G, 'beta') 
    
    # Vasomotion R_0 = R_0(1+epsilon*sin(k*s-w*t))
    R1 = R1_0*(1+eps_*ufl.sin(k_*s_-w_*t_))

    # Source term due to arterial wall motion
    f = 2.0*np.pi*R1*ufl.diff(R1, t_) # source term
    
    Res = water_properties['nu']*resistance_wrt_radius(R1/R1_0, beta)/R1_0**4
    Ainv = 1/(np.pi*(R2**2-R1_0**2))
    
    # Force term in hydraulic model
    g = (water_properties['nu']*Ainv)*ufl.diff(f, s_) 

    return f, Ainv, Res, g, t_, k_, w_, s_, eps_, R1




def resistance_wrt_radius(ra, b, c=0):
    '''
    Compute annular resistance using (10)
    
    Args: 
        ra (df.function): non-dimensionalized inner radius, ra = R1/R1_0 
        b (df.function): non-dimensionalized outer radius, b = R2/R1_0
        c (df.function): displacement of inner circle from center of annulus
        
    Returns:
        resistance (df.function): resistance of annulus
    '''
    
    # compute "permeability" kappa
    kappa = (np.pi/8)*(1+1.5*(c/(b-ra))**2 ) * (b**4-ra**4- (b**2-ra**2)**2/(ln(b/ra)) )
    
    resistance = kappa**(-1) # inverse of permeability is resistance
    return resistance