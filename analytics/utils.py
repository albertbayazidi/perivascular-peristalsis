
from graphnics import *
from xii import *
import numpy as np
def number_of_edges_in_tree(N):
    return 2**N-1


def get_lamda(k):
    # get wave length from wave number: lamda = 2*pi/k
    lamda = 2*np.pi/k
    return lamda

def get_k(lamda):
    # get wave number from wave length: k = 2*pi/lamda
    k = 2*np.pi/lamda
    return k

def get_freq(w):
    # get frequency from angular frequency: freq = w/(2*pi)
    freq = w/(2*np.pi)
    return freq

def get_w(freq):
    # get angular frequency from frequency: w = 2*pi*freq
    w = 2*np.pi*freq
    return w   


def dimensional_Q(Q, k, w, eps, radius0):
    '''
    Compute (dimensional) flow
    
    Args:
        Q (float): non-dimensionalized flow
        k (float): wave number
        w (float): angular frequency
        epsilon (float): amplitude of vasomotion
        radius0 (float): radius of vessel at rest
        
    Returns:
        dimensional flow
    '''
    
    return Q*2.0*np.pi*eps*w*radius0**2/k

def dimensional_P(P, k, w, eps, radius0, mu, rho):
    '''
    Compute (dimensional) pressure
    
    Args:
        Q (float): non-dimensionalized flow
        k (float): wave number
        w (float): angular frequency
        epsilon (float): amplitude of vasomotion
        radius0 (float): radius of vessel at rest
        mu (float): fluid viscosity
        rho (float): fluid density
        
    Returns:
        dimensional pressure
    '''
    
    return P*2.0*np.pi*mu*eps*w/(radius0**2*k**2*rho)
