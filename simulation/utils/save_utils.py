from os.path import join, isdir
from os import makedirs
from simulation.utils.save_csv_utils import make_csv
import sys
import csv
   

def singlevessel_file_name(*,freq,lambdas):
    if len(freq) > 1 and len(lambdas) == 1:
        postfix = "_".join(str(num) for num in freq)
        filename = f"exp_locled_lambda_freq_{postfix}.csv"
        return filename
    elif len(freq) == 1 and len(lambdas) > 1:
        postfix = "_".join(str(num) for num in lambdas)
        filename = f"exp_locled_freq_lambda_{postfix}.csv"
        return filename
    else:
        print("Error: Something went wrong!", file=sys.stderr)
        sys.exit(1) 

def tandemvessel_file_name():
    return "lmao"

def Yvessel_file_name():
    return "lmao"

def my_save_fun(*,Q_avg_numerical,Q_avg_analytical,Q_avg_analytical_new_imp, args):
    
    if len(args.betas) == 1:
        path_name = join("results","comparison","singlevessel")
        file_name = join(path_name,singlevessel_file_name(freq=args.freq, lambdas=args.lambdas))

    elif len(args.betas) == 2:
        path_name = join("results","comparison","tandemvessel")
        file_name = join(path_name,tandemvessel_file_name())

    elif len(args.betas) == 3:
        path_name = join("results","comparison","Yvessel")
        file_name = join(path_name,Yvessel_file_name())

    else:
        print("Error: Something went wrong!", file=sys.stderr)
        sys.exit(1) 

    if isdir(f"{path_name}") == False: 
        makedirs(f"{path_name}",exist_ok=True)
   
    data = make_csv(Q_avg_numerical, Q_avg_analytical, Q_avg_analytical_new_imp, args)

    with open(f"{file_name}", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile) 
        writer.writerows(data)

if __name__ == "__main__":
    # DEBUGGING
    
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

    args = args.parse_args()

    print(singlevessel_file_name(freq = args.freq, lambdas = args.lambdas))


