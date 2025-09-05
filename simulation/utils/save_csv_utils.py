import numpy as np
import sys

decimals = 3

def make_csv(Q_avg_numerical, Q_avg_analytical, Q_avg_analytical_new_imp, args):
    # will need more then just one since thinkgs are handled diffrenlty in the diffrent tables

    if len(args.lambdas) == 1 and len(args.freq) > 1:
        n = len(args.freq)
        freq = np.atleast_1d(args.freq)
        lambdas = np.full(n, args.lambdas)

    elif len(args.lambdas) > 1 and len(args.freq) == 1:
        n = len(args.lambdas)  
        lambdas = np.atleast_1d(args.lambdas)
        freq = np.full(n, args.freq)

    else:
        print("Error: Something went wrong!", file=sys.stderr)
        sys.exit(1) 


    Q_num = np.atleast_1d(Q_avg_numerical)
    Q_an = np.atleast_1d(Q_avg_analytical)
    Q_an_imp = np.full(n, 0) # this must be changed when Q_an_imp has been written

    rows = []
    for i in range(n):
        row = [
            i,
            args.ts_per_cycle,
            args.eps,
            args.betas,
            args.Ls,
            args.radius0,
            lambdas[i],
            freq[i],
            f"{Q_num[i]:.{decimals}e}",
            f"{Q_an[i]:.{decimals}e}",
            f"{Q_an_imp[i]:.{decimals}e}"
        ]
        rows.append(row)

    header = ["#", "delta t", "eps", "beta", "L", "r_0", "lambda", "freq", "Q_num", "Q_analyt", "Q_avg_analyt_new_imp"]
    data = [header] + rows
    return data
