from analytics.utils import *
from simulation.run_interaction_simulation import interacting_peristalsis

def run_interacting_peristalsis_simulation(G, lamdas, lambda2, freqs, freqs2, n_cycles, tsteps_per_cycle, eps, eps2):
    """
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
    """

    G.compute_edge_lengths()
    lengths = [G.edges()[e]["length"] for e in G.edges()]
    total_length = sum(lengths)

    # We make a list of "experiments" containing dicts that stores parameters and results
    experiments = []
    for lamda in lamdas:
        for freq in freqs:
            for i, tstep_per_cycle in enumerate(tsteps_per_cycle):
                params = {
                    "k": get_k(lamda),
                    "w": get_w(freq),
                    "epsilon": eps,
                    "freq": freq,
                    "lamda": lamda,
                    "l": get_k(lamda) * total_length,
                    "ts_per_cycle": tstep_per_cycle,
                    "n_cycles": n_cycles[i],
                    "T": n_cycles[i] / freq,
                }

                experiments.append(params)

    for exp in experiments:
        print("Solving...")

        # adjust mesh size to <1/200 of the wave length
        i = 3
        G.make_mesh(i)
        while G.mesh.hmax() > exp["lamda"] / 100:
            G.make_mesh(i)
            i += 1

        lamda = exp["lamda"]
        print(f"lambda={lamda}, number of mesh refinements: {i}")

        f, area_inv, res, g, t_, k1_, k2_, w1_, w2_, s_, eps1_, eps2_, R1 = interacting_peristalsis(G)

        for e in G.edges():
            G.edges()[e]["Res"] = res
            G.edges()[e]["Ainv"] = area_inv

        t_.assign(0)
        k1_.assign(get_k(lamdas[0]))
        k2_.assign(get_k(lambda2))
        
        w1_.assign(get_w(freqs[0]))
        w2_.assign(get_w(freqs2))
        
        eps1_.assign(eps)
        eps2_.assign(eps2)
        
        n_cycles = exp["n_cycles"]
        time_steps = exp["ts_per_cycle"] * n_cycles

        model = TimeDepHydraulicNetwork(G, p_bc=Constant(0), f=f, Ainv=area_inv, Res=res, g=g) 

        qps = time_stepping_stokes(model, t=t_, qp0=None, t_steps = time_steps, T=exp["T"])

        exp["sol"] = qps

    return experiments

