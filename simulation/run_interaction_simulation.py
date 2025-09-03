from graphnics import *
import networkx as nx
from xii import *
import seaborn as sns
import numpy as np
sns.set()
import matplotlib.pyplot as plt


from analytics import pvs_network_netflow *
from simulation.peristalsis import *
from analytics.utils import *


from peristalsis import *

def interacting_peristalsis(G):
    """
    Create a vasomotion model for a travelling wave
        R_0 = R_0(1+epsilon*sin(k*s-w*t))
    
    The function returns the ufl expressions needed to implement this
    in a network model
    
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
    
    k1_, k2_ = Constant(1), Constant(1)
    w1_, w2_ = Constant(1), Constant(1)
    eps1_, eps2_ = Constant(0.1), Constant(0.1)
    
    # Radius at rest
    R1_0 = nxgraph_attribute_to_dolfin(G, 'radius1') # inner radius at rest
    R2 = nxgraph_attribute_to_dolfin(G, 'radius2') # outer radius, constant in time
    beta = nxgraph_attribute_to_dolfin(G, 'beta') 
    
    # Vasomotion R_0 = R_0(1+epsilon1*sin(k1*s-w1*t)) + R_0(1+epsilon2*sin(k2*s-w2*t))
    R1 = R1_0*(1+eps1_*sin(k1_*s_-w1_*t_)) + R1_0*(1+eps2_*sin(k2_*s_-w2_*t_))

    # Source term due to arterial wall motion
    f = 2.0*np.pi*R1*ufl.diff(R1, t_) # source term
    
    
    nu = water_properties['nu']
    Res = nu*resistance_wrt_radius(R1/R1_0, beta)/R1_0**4
    
    # Inverse of area
    Ainv = 1/(np.pi*(R2**2-R1**2))   
    
    # Force term in hydraulic model
    g = 0

    return f, Ainv, Res, g, t_, k1_, k2_, w1_, w2_, s_, eps1_, eps2_, R1



def interacting_waves(L, radius0, lamdas, freqs, beta=3, n_cycles=5,tsteps_per_cycle=40, eps=0.1, folder_name='plots', kargs=None, arterial_tree=False):
    '''
    Simulate pulsatile flow due to vasomotion in a single vessel and compare with analytic solution
    
    Args:
        N (int): number of branches in the arterial tree
        L (float): length of first branch in arterial tree
        radius0 (float): radius of first branch in arterial tree
        lamdas (list): list of wave lengths
        freq (float): frequency of vasomotion
        beta (float): aspect ratio of R1 vs R2 (so R2 = beta*R1)
        n_cycles (int): number of cycles to simulate
        tsteps_per_cycle (int): number of time steps per cycle
        epsilon (float): amplitude of vasomotion
        vary_res (bool): if True, the resistance is varied in time
        vary_area (bool): if True, the cross sectional area is varied in time
    
    Returns:
        experiments (list): list of dicts containing parameters and results
        
    '''
    
    import json
    # save command line arguments 
    with open(f'{folder_name}/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    
    if arterial_tree:
        import sys
        sys.path.append('../graphnics/data/')
        from generate_arterial_tree import make_arterial_tree
        signs = np.tile([-1,1], 10).tolist()
        signs[0]=1
        signs[3]=1
        signs[4]=1
        G = make_arterial_tree(6, directions = signs, gam=0.8, L0=L, radius0=radius0)
    
    else:
    
        G = line_graph(n=2, dim=2, dx=L)
        for ix, e in enumerate(G.edges()):
            G.edges()[e]['radius'] = radius0
    
    
    for e in G.edges():
        radius0 = G.edges()[e]['radius']
        G.edges()[e]['radius1'] = radius0
        G.edges()[e]['radius2'] = radius0*beta    
        G.edges()[e]['beta'] = beta
        
    G.make_mesh(4)
    
    G.compute_edge_lengths()
    
    # time parameters
    time_steps = tsteps_per_cycle*n_cycles
    
    # Vasomotion model is expressed via ufl functions depending
    # on constant t_, k_, w_ and s_
    f, area_inv, res, g, t_, k1_, k2_, w1_, w2_, s_, eps1_, eps2_, R1 = interacting_peristalsis(G)

    for e in G.edges():
        G.edges()[e]['Res'] = res
        G.edges()[e]['Ainv'] = area_inv

    
    print('Solving...')
    t_.assign(0)
    k1_.assign(get_k(lamdas[0]))
    k2_.assign(get_k(lamdas[1]))
    
    w1_.assign(get_w(freqs[0]))
    w2_.assign(get_w(freqs[1]))
    
    eps1_.assign(eps[0])
    eps2_.assign(eps[1])
    
    model = TimeDepHydraulicNetwork(G, p_bc=Constant(0), f=f, Ainv=area_inv, Res=res, g=g, degree=2)

    T = n_cycles/min(freqs)
    print(f'Time={T}, time_steps={time_steps}')

    qps = time_stepping_stokes(model, t=t_, qp0=None, t_steps = time_steps, T=T)
    
    
    # Plot the results
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10,10)) 
    sns.set(font_scale=2.2)
        
    

    table_str = f' $\omega$ &  $\lambda$     &   $\langle Q\' \\rangle$    &    \langle Q_h\' \\rangle   \\\\ \n'
    
    
    # Compute expected net flow
    L = sum(nx.get_edge_attributes(G, 'length').values()) #total length of vessel
    
    
    # Compute net flow from simulation
    node = list(G.nodes())[-1]
    pos = G.nodes()[node]['pos']
    outflows = [sol[0](pos) for sol in qps]

    # plot over time
    time_steps = len(qps)
    dt = T/time_steps
    
    ys = np.cumsum(outflows[tsteps_per_cycle:])*dt
    ts = np.linspace(0, T, time_steps)[tsteps_per_cycle:]
    
    ## compute simulated net flow rate
    
    # we compute using the last cycle, need to know how many time steps to slice through
    T_cycle = 1/min(freqs)
    tsteps_per_cycle_for_this_freq = int(T_cycle/dt)
    
    Qh_avg_tilde = (ys[-1]-ys[-tsteps_per_cycle_for_this_freq-1])/T_cycle
   
   
    ax.plot(ts, ys, linewidth=3)
    
    plt.subplots_adjust(hspace=0.2)
    ax.set_ylabel('$\int_0^{t\'} Q\'(\\tau) \, \mathrm{d} \\tau$ [$\mu$L]', fontsize=24)
    ax.set_xlabel('t\' [s]', fontsize=24)
    
    # Save table and figure
    
    # Description of the geometry and vasomotion    
    freqs_str = '-'.join([f'{freq:g}' for freq in freqs])
    lamdas_str = '-'.join([f'{l:1.2f}' for l in lamdas])
    eps_str = '-'.join([f'{eps:1.2f}' for eps in eps])
    vasomotion_desc = f'eps{eps_str}_freq{freqs_str}_lamda{lamdas_str}'
    
    # Now save    
    plt.savefig(f'{folder_name}/netflow_{vasomotion_desc}.png', bbox_inches="tight")
    
    # Also save the solutions
    h5_q = HDF5File(G.mesh.mpi_comm(), f'{folder_name}/sols/q_{vasomotion_desc}.h5', 'w')
    h5_p = HDF5File(G.mesh.mpi_comm(), f'{folder_name}/sols/p_{vasomotion_desc}.h5', 'w')
    for ix, qp in enumerate(qps[1:]):
        h5_q.write(qp[0], 'q', ix)
        h5_p.write(qp[1], 'p', ix)
    h5_q.close()
    h5_p.close()
    
    
    # Save graph
    G_nx = nx.DiGraph(G)
    for e in G_nx.edges():
        del G_nx.edges()[e]['Ainv']
        del G_nx.edges()[e]['Res']
    nx.write_gpickle(G_nx, folder_name + "/G.gpickle")
    
if __name__ == '__main__':
    
    import argparse as arg
    args = arg.ArgumentParser()
    
    args.add_argument('--L', type=float, default=1)
    args.add_argument('--radius0', type=float, default=0.1)
    
    args.add_argument('--lambdas', type=float, nargs='+', default=[100, 4]) # 
    args.add_argument('--freq', type=float, nargs='+', default=[10, 0.1]) # 
    args.add_argument('--epsilon', type=float, nargs='+', default=[0.01, 0.1])
    args.add_argument('--n_cycles', type=int, default=20)
    args.add_argument('--tsteps_per_cycle', type=int, default=1000)
    
    args.add_argument('--tree', type=int, default=0)
    
    args = args.parse_args()
    
    folder_name = 'results/interacting'
    
    if os.path.isdir(f'{folder_name}') == False: #in case the folder doesn't exist
        os.mkdir(f'{folder_name}')
    
    interacting_waves(L=args.L, radius0=args.radius0, eps=args.epsilon, lamdas=args.lambdas, freqs=args.freq, n_cycles=args.n_cycles, tsteps_per_cycle=args.tsteps_per_cycle, folder_name=folder_name, kargs=args, arterial_tree=args.tree)
    
        




