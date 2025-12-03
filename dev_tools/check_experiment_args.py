import sys

def check_args(args):
    betas = args.betas
    Ls = args.Ls
    depth = args.depth[0]
    r0 = args.radius0
    lambdas = args.lambdas
    freqs = args.freq
    n_cycles = args.n_cycles
    ts_per_cycle = args.ts_per_cycle
    eps = args.eps

    if n_cycles == 0: n_cycles = ts_per_cycle

    if len(betas) != len(Ls):
        print("Error: The number of 'betas' and 'Ls' values must be equal.")
        sys.exit(1)
    
    if depth >= 3:
        if len(Ls) > 1:
            print("Error: More then one value is incorrect for trees over 2 generations.")
            sys.exit(1)

    if depth == 2:
        if len(betas) != 3 or len(Ls) != 3:
            print("Error: Depth 2 requires exactly 3 'betas' and 3 'Ls'.")
            sys.exit(1)

    if depth == 1:
        if not (len(betas) in [1, 2] and len(Ls) in [1, 2]):
            print("Error: Depth 1 requires 1 or 2 'betas' and 'Ls' depending on experiment type.")
            sys.exit(1)

    return betas, Ls, depth, r0, lambdas, freqs, n_cycles, ts_per_cycle, eps
