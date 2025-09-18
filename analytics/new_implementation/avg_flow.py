from analytics.new_implementation.helper_functions import dimensional_Q
from analytics.new_implementation.volemetric_flow import *

def experiment_parameters(exp, Ls):

    k, w  = [exp[key] for key in ['k', 'w']]

    ls = [k*L for L in Ls] 

    return k, w, ls


def avg_flow(G, experiments, radius0, betas, Ls, eps ):
   
    Q_array = [] # should be length of nr of experiments

    for _, exp in enumerate(experiments):
    
        # Grab parameters 
        k, w, ls = experiment_parameters(exp, Ls) 
        
        Q_dimless_array = get_Q(radius0, betas, ls, eps, G)

        Q_dimensioned  =  [dimensional_Q(Q, k, w, eps, radius0) for Q in Q_dimless_array][0] # only keeping root node value
        
        Q_array.append(Q_dimensioned)


    return Q_array

