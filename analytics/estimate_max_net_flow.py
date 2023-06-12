from sympy import *
import math

# Define f 
l = Symbol("l")
f = 1/l*(1./2 - (1 - cos(l))/(l*l))

# Differentiate f with respect to l
dfdl = diff(f, l)

# Find l_0 such that df/dl(l_0) = 0
guess = 3
l_0 = nsolve(dfdl, l, guess)
print("l_0 = ", l_0)

# Evaluate f(l_0):
f_0 = f.subs(l, l_0)
print("f_0 = ", f_0)

# Max wave length
L = Symbol("L") # m
lmbda_0 = 2*math.pi*L/l_0
print("\lambda_0 = ", lmbda_0)

# Evaluate peak net flow Q_0 (need numbers)
epsilon = Symbol("eps") # m 
r_0 = Symbol("r_0") # m
Delta = Symbol("Delta") # m^{-2}
omega = Symbol("omega") # 1 / 2
S = 2*math.pi*f_0
print("S = ", S)
Q_0 = epsilon**2*omega*r_0**2*L*Delta*S
print("Q_0 = ", Q_0, "(m)^3 / s")

