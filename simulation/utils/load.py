from graphnics import *
import networkx as nx
import h5py
import pickle

def load_raw_data(exp_folder, pressure_path, flux_path, exp_data_file):
    # DET ER FORSKJELLIGE MESH NÅR L endrer seg
    with open(f"{exp_data_file}", "rb") as f:
        loaded_exp_data = pickle.load(f)
    
    G = nx.read_gpickle(f"{exp_folder}/G.gpickle")
    G = copy_from_nx_graph(G)

    i = 3
    G.make_mesh(i)
    while G.mesh.hmax() > loaded_exp_data["lamda"] / 100:
        G.make_mesh(i)
        i += 1

    V = FunctionSpace(G.mesh, "DG", 0)
    M = FunctionSpace(G.mesh, "CG", 1)
    
    qps = []
    with h5py.File(flux_path, "r") as h5_q, h5py.File(pressure_path, "r") as h5_p:

        count = int(h5_q["flux/iteration"][()])
        for i in range(count):
            q = Function(V)
            q.vector()[:] = h5_q[f"flux/vector_{i}"][()]

            p = Function(M)
            p.vector()[:] = h5_p[f"pressure/vector_{i}"][()]

            qps.append((q.copy(deepcopy=True), p.copy(deepcopy=True)))

    return G, qps, loaded_exp_data 
    

