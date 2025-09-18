from graphnics import *
import argparse as argp
import numpy as np

import sys
sys.path.append('../graphnics/data/')
from generate_arterial_tree import make_arterial_tree

from dev_tools.make_exp_dict import make_exp_dict
from simulation.numerical_bifurcation import run_numerics
from analytics.new_implementation.avg_flow import avg_flow
from analytics.new_implementation.utils.visualization_utils import plot_tree

from simulation.utils.save import make_experiment_result_dict,save_raw_data 

def compare(G, r0, Ls, betas, lamdas, freqs, n_cycles=0, ts_per_cycle=40, eps=0.1):
    '''
    Simulate pulsatile flow due to vasomotion in bifurcated vessel (of n generations) and compare with analytic solution
    
    Args:
        r0 (float): inner radius of vessels at rest
        lamdas (list): list of wave lengths
        freq (float): frequency of vasomotion
        betas (list): aspect ratios of R1 vs R2 (so R2 = beta*R1)
        Ls (list): vessel lengths
        n_cycles (int): number of cycles to simulate
        tsteps_per_cycle (int): number of time steps per cycle
        eps (float): amplitude of vasomotion
        
    '''

    Q_avg_num_results, experiments = run_numerics(G, lamdas, freqs, n_cycles, ts_per_cycle, eps)
    
    
    Q_avg_analytical = avg_flow(G, experiments, r0, betas, Ls, eps)

    return Q_avg_num_results, Q_avg_analytical, experiments

if __name__ == "__main__":
    
    args = argp.ArgumentParser()
    
    # domain parameters
    args.add_argument("--beta0", nargs="+", type=float, default=2)
    args.add_argument("--L0", nargs="+", type=float, default=1)
    args.add_argument("--radius0", type=float, default=0.1)
    args.add_argument("--gamma", type=float, default=0.8)

    # peristalsis parameters
    args.add_argument("--lambdas", type=float, nargs="+", default=[1])
    args.add_argument("--freq", type=float, nargs="+", default=[1]) 
    args.add_argument("--eps", type=float, default=0.1)
    
    # numerical parameters
    args.add_argument("--ts_per_cycle", nargs="+", type=int, default=[10])
    args.add_argument("--n_cycles",  nargs="+", type=int, default=0)


    # exstra parameters
    args.add_argument("--tree", help="Constructs of a tree with n generations", nargs="+", type=int, default=2)
    args.add_argument("--plot", action="store_true", help="Enable plotting")

    args = args.parse_args()
    
    ## ERROR HANDLING
    if args.tree[0] < 3:  sys.exit("Error: please run run_comparison for only two generation")
    
    if args.n_cycles == 0:
        args.n_cycles = args.ts_per_cycle
    
    beta0 = args.beta0
    r0 = args.radius0
    lambdas = args.lambdas
    freqs = args.freq
    n_cycles = args.n_cycles
    ts_per_cycle = args.ts_per_cycle 
    eps = args.eps

    signs = np.tile([-1,1], 10).tolist()
    signs[0]=1
    signs[3]=1
    signs[4]=1

    G = make_arterial_tree(args.tree[0], directions = signs, gam=0.8, L0=args.L0, radius0=r0)

    for e in G.edges():
        radius0 = G.edges()[e]['radius']
        G.edges()[e]['radius1'] = radius0
        G.edges()[e]['radius2'] = radius0*beta0    
        G.edges()[e]['beta'] = beta0
        
    G.make_mesh(1) 
    G.compute_edge_lengths()
    
    Ls = [G.edges()[e]['length']for e in G.edges()]
    betas = np.full(len(Ls),beta0)

    if args.plot: plot_tree(G) 

    Q_avg_num_results, Q_avg_analytical, experiments = compare(G, r0, Ls, betas, lambdas,
                                                                       freqs, n_cycles, ts_per_cycle, eps)

    exp_result_dict = make_experiment_result_dict(r0, Ls, beta0, experiments, Q_avg_num_results, Q_avg_analytical)
    save_raw_data(exp_result_dict);

