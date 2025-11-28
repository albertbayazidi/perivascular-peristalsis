from dev_tools.ignore_warning_cbc_block import *
from graphnics import *
import argparse as argp
import numpy as np

from graph.choose_graph import setup_graph
from graph.visualize_tree import plot_tree

from dev_tools.make_exp_dict import make_dummy_exp_results
from dev_tools.check_experiment_args import check_args

from simulation.numerical import run_numerics
from analytics.new_implementation.avg_flow import avg_flow

from simulation.utils.save import make_experiment_result_dict,save_raw_data

def compare(G, r0, Ls, betas, lamdas, freqs, n_cycles=0, ts_per_cycle=40, eps=0.1):
    """
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

    """

    Q_avg_num_results, experiments = run_numerics(G, lamdas, freqs, n_cycles, ts_per_cycle, eps)
    #Q_avg_num_results, experiments = make_dummy_exp_results(G, lamdas, freqs, n_cycles, ts_per_cycle, eps) # Debugging tool

    Q_avg_analytical = avg_flow(G, experiments, r0, betas, Ls, eps)

    return Q_avg_num_results, Q_avg_analytical, experiments


def main():
    args = argp.ArgumentParser()
    
    # domain parameters
    args.add_argument("--betas", nargs="+", type=float, default=[2])
    args.add_argument("--Ls", nargs="+", type=float, default=[1])
    args.add_argument("--radius0", type=float, default=0.1)
    args.add_argument("--gamma", type=float, default=0.8)

    # peristalsis parameters
    args.add_argument("--lambdas", type=float, nargs="+", default=[1])
    args.add_argument("--freq", type=float, nargs="+", default=[1])
    args.add_argument("--eps", type=float, default=0.1)

    # numerical parameters
    args.add_argument("--ts_per_cycle", nargs="+", type=int, default=[10])
    args.add_argument("--n_cycles", nargs="+", type=int, default=0)

    # exstra parameters
    args.add_argument("--depth", help="Constructs of a tree with n generations", nargs="+", type=int, default=1)
    args.add_argument("--plot", action="store_true", help="Enable plotting")

    args = args.parse_args()

    betas, Ls, depth, r0, lambdas, freqs, n_cycles, ts_per_cycle, eps = check_args(args)

    G = setup_graph(depth, betas, Ls, r0)

    if depth > 2:
        Ls = [G.edges()[e]["length"] for e in G.edges()]
        betas = np.full(len(Ls), betas[0])

    if args.plot: plot_tree(G)

    Q_avg_num_results, Q_avg_analytical, experiments = compare(G, r0, Ls, betas, lambdas, freqs, n_cycles, ts_per_cycle, eps)

    exp_result_dict = make_experiment_result_dict(depth, r0, Ls, betas, experiments, Q_avg_num_results, Q_avg_analytical,G)

    _, json_path = save_raw_data(exp_result_dict, G, experiments, 1, 1)

    return json_path


if __name__ == "__main__":
    json_path = main()

    print(json_path)
