import numpy as np

def get_k(lamda):
    # get wave number from wave length: k = 2*pi/lamda
    k = 2*np.pi/lamda
    return k

def get_w(freq):
    # get angular frequency from frequency: w = 2*pi*freq
    w = 2*np.pi*freq
    return w   

def _invR(beta):
    "Helper function for evaluating \\mathcal{R}^{-1}(beta)."
    val = np.pi/8*(beta**4 - 1 - (beta**2 - 1)**2/np.log(beta))
    return val

def _R(beta):
    "Helper function for evaluating \\mathcal{R}(beta)."
    return 1.0/_invR(beta)

def _delta(beta):
    "Helper function for evaluating Delta(beta)."
    delt = ((2 - (beta**2-1)/np.log(beta))**2)/(beta**4 - 1 - (beta**2 - 1)**2/np.log(beta))
    return delt

def _beta(r_e, r_o):
    return r_e/r_o

def _alpha(l, P, R):
    # eq 34 uten deltar og første ledd
    "Helper function for matrix/vector expression."
    z = 1j
    A1 = (0.5 - (1 - np.cos(l))/(l**2))
    A2 = P*(1 - np.exp(z*l))/(2*l**2*R)
    return (A1 + A2.real)

def _xi(l):
    "Helper function for matrix/vector expression."
    z = 1j
    xi1 = (np.exp(z*l) - 1)/l
    return xi1

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

