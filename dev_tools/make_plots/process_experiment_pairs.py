import os
import sys
import argparse
from simulation.utils.load import load_raw_data
from simulation.utils.plot_net_flow import save_net_flow_at_nodes
from simulation.utils.plot_velocity_field import save_velocity_at_nodes

def process_experiment_pair(non_rem_path, rem_path, plot_window, target_nodes=None):
    
    graph_path = non_rem_path 

    pressure_path_rem = os.path.join(rem_path, "pressure", "HDF5", f"sols_0.h5")
    flux_path_rem = os.path.join(rem_path, "flux", "HDF5", f"sols_0.h5")
    exp_data_file_path_rem = os.path.join(rem_path, "exp_data", f"exp_0.pkl")

    pressure_path_non_rem = os.path.join(non_rem_path, "pressure", "HDF5", f"sols_0.h5")
    flux_path_non_rem = os.path.join(non_rem_path, "flux", "HDF5", f"sols_0.h5")
    exp_data_file_path_non_rem = os.path.join(non_rem_path, "exp_data", f"exp_0.pkl")

    _, qps_rem, exp_data_rem = load_raw_data(graph_path, pressure_path_rem,
                                             flux_path_rem, exp_data_file_path_rem)

    G, qps_non_rem, exp_data_non_rem = load_raw_data(graph_path, pressure_path_non_rem,
                                                     flux_path_non_rem, exp_data_file_path_non_rem)

    # Save/Process Results
    save_velocity_at_nodes(G, qps_non_rem, qps_rem, exp_data_non_rem, exp_data_rem,  
                           non_rem_path, rem_path, 
                           plot_window=plot_window, target_node_indices=target_nodes)

    save_net_flow_at_nodes(G, qps_non_rem, qps_rem, exp_data_non_rem, exp_data_rem, 
                           non_rem_path, rem_path, 
                           plot_window=plot_window, target_node_indices=target_nodes)

if __name__ == "__main__":
    """
    example command python process_experiment.py ./non_rem_folder ./rem_folder --window 50 150 --nodes 1 4 10
    """

    parser = argparse.ArgumentParser(description="Process REM and Non-REM experiment data.")
    parser.add_argument("non_rem_path", type=str, help="Path to the Non-REM experiment folder")
    parser.add_argument("rem_path", type=str, help="Path to the REM experiment folder")
    parser.add_argument("--window", type=int, nargs='+', default=-1, help="Plot window.")
    parser.add_argument("--nodes", type=int, nargs='+', default=0, help="target node indices")
   
    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.non_rem_path):
        print(f"Error: Non-REM path not found: {args.non_rem_path}")
        sys.exit(1)
    if not os.path.exists(args.rem_path):
        print(f"Error: REM path not found: {args.rem_path}")
        sys.exit(1)
    
    print(f"NON-REM: {args.non_rem_path}")
    print(f"REM:     {args.rem_path}")

    # Run the main function
    process_experiment_pair(
        rem_path=args.rem_path, 
        non_rem_path=args.non_rem_path, 
        plot_window=args.window,
        target_nodes=args.nodes)

