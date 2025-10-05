from analytics.utils import get_k, get_w
import numpy as np

def make_dummy_exp_results(G, lamdas, freqs, n_cycles, ts_per_cycle, eps):
    dummy_num_results = np.full(len(lamdas) * len(freqs) * len(ts_per_cycle) ,0)
    G.compute_edge_lengths()
    lengths = [G.edges()[e]['length'] for e in G.edges()]
    total_length = sum(lengths)
    
    experiments = []
    for lamda in lamdas:
        for freq in freqs:
            for i, tstep_per_cycle in enumerate(ts_per_cycle):
                
                params = {'k':get_k(lamda), 
                          'w':get_w(freq), 
                          'epsilon':eps, 
                          'freq':freq, 
                          'lamda':lamda, 
                          'l':get_k(lamda)*total_length, 
                          'ts_per_cycle':tstep_per_cycle,
                           'n_cycles':n_cycles[i], 
                           'T':n_cycles[i]/freq}
                
                experiments.append(params)
    
    return dummy_num_results, experiments
