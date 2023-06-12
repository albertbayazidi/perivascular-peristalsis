# Run with
# $ python3 estimate_delta.py

from numpy import log, pi

# Parameters
r_o = 7 # micron
print("r_o = ", r_o)
r_e = 12 # micron
print("r_e = ", r_e)
varepsilon = 0.1 # unitless
print("varepsilon = ", varepsilon)

L_opt = 2550 # micron
print("L_opt (micron) = ", L_opt)
freq = 0.1  # Hz = 1/s
print("freq (s) = ", freq)

# Definition of beta
beta = r_e/r_o
print("beta = ", beta)

# Definition of Delta
Delta = ((2 - (beta**2-1)/log(beta))**2)/(beta**4 - 1 - (beta**2 - 1)**2/log(beta))
print("Delta = ", Delta)

# Definition of \omega
omega = 2*pi*freq

# Computed peak f
f_max = 0.09916

# Estimate for max Q
Q_max = 2*pi*varepsilon**2*omega*r_o**2*L_opt*Delta*f_max

print("Q_max (microm^3/s) = %e" % Q_max)
print("Q_max (mm^3/s) = %e" % (Q_max*1.e-9))

area = pi*(r_e**2 - r_o**2)

print("area (microm^2) = %e" % area)
print("v_max (microm/s) = ", Q_max/area)
