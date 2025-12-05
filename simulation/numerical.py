import numpy as np

from simulation.peristalsis import *
from simulation.interacting_peristalsis import * 

def run_numerics(G, lamdas, freqs, n_cycles=0, ts_per_cycle=40, eps=0.1):
    """
    Simulate pulsatile flow due to vasomotion in a bifurcated vessel

    Args:
        G (nx.Graph): graph representing the network
            - Lengths
            - Radius1 inner
            - Radius2 outer
            - Betas
        lamdas (list): list of wave lengths
        freq (float): frequency of vasomotion
        n_cycles (int): number of cycles to simulate
        tsteps_per_cycle (int): number of time steps per cycle
        eps (float): amplitude of vasomotion

    """
    state = "interacting"
    if state == "interacting":
        u, v = list(G.edges())[0]
        R0 = G.edges[u, v]["radius1"]
        cardiac_wave_speed = 3200
        if (R0 == 0.006):                                       # NONREM
            freqs2 = 8                                          # Hz
            lambda2 = np.round(cardiac_wave_speed/freqs2)       # mm
            eps2 = 0.0001625/R0 
        else:                                                   # REM 
            freqs2 = 11.11                                      # Hz
            lambda2 = np.round(cardiac_wave_speed/freqs2)       # mm
            eps2 = 0.000125/R0

        experiments = run_interacting_peristalsis_simulation(G=G, lamdas=lamdas, lambda2=lambda2,
                                                             freqs=freqs, freqs2=freqs2, n_cycles=n_cycles,
                                                             tsteps_per_cycle=ts_per_cycle, eps=eps, eps2=eps2)
        
        dummy_num_results = np.full(len(lamdas) * len(freqs) ,0)
        return dummy_num_results, experiments


    # Run experiments and get results
    experiments = run_peristalsis_simulation(G=G, lamdas=lamdas, freqs=freqs, 
                                        n_cycles=n_cycles, tsteps_per_cycle=ts_per_cycle, epsilon=eps)
    Q_avg_num_results = []
    
    for _, exp in enumerate(experiments):
        # Grab parameters
        n_cycles = [exp[key] for key in ["n_cycles"]][0]

        # Compute net flow from simulation
        node_positions = [G.nodes()[n]["pos"] for n in G.nodes()]

        outflows_per_node = [[sol[0](pos) for sol in exp["sol"]] for pos in node_positions]

        T = n_cycles / exp["freq"]  # total simulation time
        T_cycle = 1.0 / exp["freq"]
        total_time_steps = len(exp["sol"])
        dt = T / total_time_steps

        ts_per_cycle = exp["ts_per_cycle"]
        # integrate over last cycle to get net flow
        netflow_per_node = [np.cumsum(outflow[ts_per_cycle:]) * dt for outflow in outflows_per_node]

        netflow_root = netflow_per_node[0]

        Qh_avg_tilde_root = (netflow_root[-1] - netflow_root[-ts_per_cycle - 1]) / T_cycle

        # save Qh for each experiment
        Q_avg_num_results.append(Qh_avg_tilde_root)

    return Q_avg_num_results, experiments
