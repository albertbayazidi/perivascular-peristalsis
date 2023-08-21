from graphnics import *
from xii import *
import numpy as np

import sys
sys.path.append('../analytics')

from pvs_network_netflow import *
from peristalsis import *
from utils import *


def run_comparison(G, radius0, lamdas, freqs, betas, Ls,
                             n_cycles=0, ts_per_cycle=40, eps=0.1, 
                             folder_name='results', kargs=None):
    '''
    
    Simulate pulsatile flow due to vasomotion in a bifurcated vessel and compare with analytic solution
    
    Args:
        radius0 (float): inner radius of vessels at rest
        lamdas (list): list of wave lengths
        freq (float): frequency of vasomotion
        betas (list): aspect ratios of R1 vs R2 (so R2 = beta*R1)
        Ls (list): vessel lengths
        n_cycles (int): number of cycles to simulate
        tsteps_per_cycle (int): number of time steps per cycle
        eps (float): amplitude of vasomotion
        
    '''
    
    # Run experiments and get results
    experiments = run_peristalsis_simulation(G=G, lamdas=lamdas, freqs=freqs, 
                                        n_cycles=n_cycles, tsteps_per_cycle=ts_per_cycle, epsilon=eps)
    
    
    
    table_str = f'$\lambda$  &  \langle Q_h\' \\rangle   $\langle Q\' \\rangle$    & \\\\ \n'
    for i, exp in enumerate(experiments):
    
        # Grab parameters 
        G, k, w, n_cycles, lamda_val = [exp[key] for key in ['G', 'k', 'w', 'n_cycles', 'lamda']]
               
        
        # Compute net flow from simulation
        node_positions = [G.nodes()[n]['pos'] for n in G.nodes()]
        
        outflows_per_node = [[sol[0](pos) for sol in exp['sol']] for pos in node_positions]
        
        T = n_cycles/exp['freq'] # total simulation time
        T_cycle = 1.0/exp['freq']
        total_time_steps = len(exp['sol'])
        dt = T/total_time_steps
        
        ts_per_cycle = exp['ts_per_cycle']
        # integrate over last cycle to get net flow
        netflow_per_node = [np.cumsum(outflow[ts_per_cycle:])*dt for outflow in outflows_per_node]
        
        netflow_root = netflow_per_node[0]
        
        import matplotlib.pyplot as plt
        plt.plot(netflow_root)
        plt.savefig(f'netflow.png')
        
        Qh_avg_tilde_root = (netflow_root[-1]-netflow_root[-ts_per_cycle-1])/T_cycle
        
        
        G.compute_edge_lengths()
        ls = [k*L for L in Ls] # fractional lengths
    
        Qs = get_Q(radius0, betas, ls, eps)
        
        Qs_tilde = [dimensional_Q(Q, k, w, eps, radius0) for Q in Qs]
        
        Q_tilde_root = Qs_tilde[0]
        
        rel_error = np.abs(Q_tilde_root-Qh_avg_tilde_root)/Q_tilde_root*100

        table_str += f'{lamda_val:g}  & {Qh_avg_tilde_root:<1.3e} ({rel_error:1.2f}\%)   & {Q_tilde_root:<1.3e} \\\\ \n' 
        
    table_str += f'\n\n\%{str(kargs)}'
    print(table_str)
    
    # Save table and figure
    if os.path.isdir(f'{folder_name}') == False: #in case the folder doesn't exist
        os.mkdir(f'{folder_name}')
    
    # Description of the geometry and vasomotion    
    geom_desc = f'{len(G.edges())}vessel'  
    vasomotion_desc = f'eps{eps:1.1f}'
    
    # Now save    
    table_file = open(f'table_{geom_desc}_{vasomotion_desc}.txt', 'w')
    table_file.write(table_str)
    table_file.close()
    
    
    
    
def get_Q(r0, betas, ls, eps):
    '''
    Get net flow for a bifurcated vessel with 3 vessels
    
    TODO: Return non-one
    '''
    
    if len(betas) == 1:
        Q = get_Q_single(betas[0], ls[0], eps)
        
        Q = [Q]
    if len(betas) == 2:
        Q = get_Q_tandem(r0, betas, ls, eps)
        Q = [Q, Q]
        
    if len(betas) == 3:
        indices = [(0, 1, 2),]
        paths = [(0, 1), (0, 2)]
                
        (P, dP, Q1s) = solve_bifurcating_tree(indices, paths, betas, ls)
        Q = [Q1*eps for Q1 in Q1s]
        
    return Q
    

def setup_graph(betas, Ls, radius0):
    '''
    Make a graph with 1, 2 or 3 vessels, with lengths Ls and aspect ratios betas
    '''
    
        
    # Set up graph and node positions
    if len(betas) == 1:
        G = line_graph(n=2, dim=2, dx=Ls[0])
        
    elif len(betas) == 2:
        La, Lb = Ls
        
        G = line_graph(n=3, dim=2, dx=La)

        G.nodes()[0]['pos'] = [0, 0]
        G.nodes()[1]['pos'] = [La, 0]
        G.nodes()[2]['pos'] = [La+Lb, 0]
        
        
    elif len(betas) == 3:
        G = Y_bifurcation()
        
        La, Lb, Lc = Ls
        G.nodes()[0]['pos'] = [0, 0]
        G.nodes()[1]['pos'] = [0, La]
        G.nodes()[2]['pos'] = [-np.sqrt(0.5)*Lb, La+np.sqrt(0.5)*Lb]
        G.nodes()[3]['pos'] = [ np.sqrt(0.5)*Lc, La+np.sqrt(0.5)*Lc]
    
    else: 
        raise ValueError('Only 1, 2, or 3 vessels supported')
    
    # Add inner and outer radiuses, and beta values, as edge attributes
    nx.set_edge_attributes(G, radius0, 'radius1')
    
    # make dict of betas
    betas_dict = {e:beta for e,beta in zip(G.edges(), betas) }
    nx.set_edge_attributes(G, betas_dict, 'beta')
    
    radius2_dict = {e:beta*radius0 for e,beta in zip(G.edges(), betas) }
    nx.set_edge_attributes(G, radius2_dict, 'radius2')
    
    G.make_mesh(1) 
    
    return G
    

        
if __name__ == '__main__':
    
    import argparse as argp
    args = argp.ArgumentParser()
    
    # domain parameters
    args.add_argument('--betas', nargs='+', type=float, default=[2, 2])
    args.add_argument('--Ls', nargs='+', type=float, default=[1, 1])
    args.add_argument('--radius0', type=float, default=0.1)
    
    # peristalsis parameters
    args.add_argument('--lambdas', type=float, nargs='+', default=[1])
    args.add_argument('--freq', type=float, nargs='+', default=[1]) 
    args.add_argument('--eps', type=float, default=0.1)
    
    # numerical parameters
    args.add_argument('--ts_per_cycle', nargs='+', type=int, default=[10])
    args.add_argument('--n_cycles',  nargs='+', type=int, default=0)
    
    args = args.parse_args()
    
    if args.n_cycles == 0:
        args.n_cycles = args.ts_per_cycle
        
        
    assert len(args.betas) == len(args.Ls), 'Number of betas must equal number of Ls'
        
    G = setup_graph(args.betas, args.Ls, args.radius0)
    
    run_comparison(G, radius0=args.radius0, lamdas=args.lambdas, freqs=args.freq, betas=args.betas, Ls=args.Ls, eps=args.eps,
                             n_cycles=args.n_cycles, ts_per_cycle=args.ts_per_cycle)
