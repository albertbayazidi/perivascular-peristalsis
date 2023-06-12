from numpy import pi, log, cos, linspace, meshgrid, empty, ones

import seaborn as sns
sns.set(font_scale=2)
sns.set_style("whitegrid")
import pylab
from matplotlib.cm import ScalarMappable

def invR(beta):
    val = pi/8*(beta**4 - 1 - (beta**2 - 1)**2/log(beta))
    return val

def R(beta):
    return 1.0/invR(beta)

def delta(beta):
    delt = ((2 - (beta**2-1)/log(beta))**2)/(beta**4 - 1 - (beta**2 - 1)**2/log(beta))
    return delt
    
def beta(r_e, r_o):
    return r_e/r_o

def F(ell):

    val = 1./ell*(0.5 - (1 - cos(ell))/(ell**2))
    return val
    

def Q(beta_a, beta_b, l_a, l_b):

    delta_a = delta(beta_a) 
    delta_b = delta(beta_b)

    R_a = R(beta_a)
    R_b = R(beta_b)
    
    Rlab = R_a*l_a + R_b*l_b
   
    T1 = (delta_a*R_a*l_a + delta_b*R_b*l_b)/(2*Rlab)
    T2 = (delta_a*R_a**2*(1-cos(l_a)) + delta_b*R_b**2*(1-cos(l_b)))/(Rlab**2)
    T3 = (delta_a + delta_b)*R_a*R_b/(2*Rlab**2)*(1 - cos(l_a) - cos(l_b) + cos(l_a + l_b))
    varepsilon = 1.0
    Q = varepsilon*(T1 - T2 + T3)
    return Q

def generate_tandem_plot():


    l_as = linspace(0.0, 1.0, 501)
    l_bs = 1.0 - l_as

    beta_a = 3.0
    beta_b = 3.0
    Q_ref = Q(beta_a, beta_b, l_as, l_bs)
    Q_max_3 = Q(beta_a, beta_b, l_as, l_bs)

    beta_a = 2.0
    beta_b = 4.0
    Q_max_2 = Q(beta_a, beta_b, l_as, l_bs)

    beta_a = 4.0
    beta_b = 2.0
    Q_max_4 = Q(beta_a, beta_b, l_as, l_bs)

    beta_a = 2.0
    beta_b = 2.0
    Q_max_22 = Q(beta_a, beta_b, l_as, l_bs)

    beta_a = 4.0
    beta_b = 4.0
    Q_max_44 = Q(beta_a, beta_b, l_as, l_bs)

    l_as_1000 = linspace(0.0, 1.0, 1001)
    l_bs_1000 = 1.0 - l_as_1000
    beta_a = 2.0
    beta_b = 4.0
    Q_max_2_1000 = Q(beta_a, beta_b, l_as_1000, l_bs_1000)

    l_as_250 = linspace(0.0, 1.0, 251)
    l_bs_250 = 1.0 - l_as_250
    beta_a = 2.0
    beta_b = 4.0
    Q_max_2_250 = Q(beta_a, beta_b, l_as_250, l_bs_250)

    pylab.figure(figsize=(10,8))
    pylab.plot(l_as, Q_max_3, label="$\\beta_b = \\beta_a$", linewidth=4.0)
    pylab.plot(l_as, Q_max_2, label="$\\beta_b > \\beta_a$", linewidth=4.0)
    pylab.plot(l_as, Q_max_4, label="$\\beta_b < \\beta_a$", linewidth=4.0)

    verify = False
    if verify:
        pylab.plot(l_as_250, Q_max_2_250, '.-', label="$\\beta_b > \\beta_a$ (coarse)", linewidth=4.0)
        pylab.plot(l_as_1000, Q_max_2_1000, '--', label="$\\beta_b > \\beta_a$ (fine)", linewidth=4.0)
        pylab.plot(l_as, Q_max_22, "--", label="$\\beta_b = \\beta_a = 2$", linewidth=4.0)
        pylab.plot(l_as, Q_max_44, "--", label="$\\beta_b = \\beta_a = 4$",  linewidth=4.0)

    pylab.ylabel("$\\langle Q \\rangle / \\varepsilon$")
    pylab.xlabel("$\\ell_a / (\\ell_a + \\ell_b)$")
    pylab.grid(False)
    pylab.tight_layout()
    pylab.legend()
    pylab.savefig("graphics/tandem_tapering.pdf")
    pylab.show()
    

def generate_contour_plots(beta_a):

    ls = linspace(1./3, 2./3, 1000)
    betas = linspace(1.5, 3.5, 1000)

    beta_bs, l_as = meshgrid(betas, ls, indexing="xy")
    #print("l_as = ", l_as)

    l_bs = 1.0 - l_as
    #print("l_b = ", l_bs)
    
    beta_as = beta_a*ones(len(beta_bs.T))
    #print("beta_as = ", beta_as)
    
    Q_ref = Q(beta_a, beta_as, l_as, l_bs)
    print("Q_ref = ", Q_ref[0][0])

    Q_max = Q(beta_a, beta_bs, l_as, l_bs)
    #print("Q_max = ", Q_max)
    
    Q_rel = Q_max/Q_ref
    #print("Q_rel = ", Q_rel)
    
    I = l_bs/l_as

    print("I.shape = ", I.shape)
    print("beta_bs.shape = ", beta_bs.shape)
    print("Q_rel.shape = ", Q_rel.shape)

    vmin = 0.5
    vmax = 1.5
    
    pylab.figure(figsize=(10, 8))
    levels = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 3.0]
    cp1 = pylab.contour(beta_bs, I, Q_rel, levels, cmap="seismic", vmin=vmin, vmax=vmax)
    pylab.clabel(cp1, inline=True, fontsize=14, fmt="%.1f")
    cp2 = pylab.contourf(beta_bs, I, Q_rel, levels, cmap=cp1.cmap, alpha=0.5,
                         vmin=vmin, vmax=vmax)

    pylab.colorbar(
        ScalarMappable(norm=cp1.norm, cmap=cp1.cmap),
        ticks=(vmin, 1.0, vmax)
    )

    #pylab.colorbar(cp1)
    pylab.ylabel('$\\ell_b/\\ell_a$')
    pylab.yticks([0.5, 1.0, 1.5, 2.0])
    pylab.xticks([1.5, 2.0, 2.5, 3.0, 3.5])
    pylab.xlabel('$\\beta_b$')
    pylab.tight_layout()
    pylab.grid(False)
    pylab.savefig("graphics/relative_increase_tandem_beta_a%g.pdf" % beta_a)
    pylab.show()

def generate_plot_of_F():
    ells = linspace(0.0, 20.0, 10000)
    fs = F(ells)
    pylab.figure(figsize=(10, 8))
    pylab.plot(ells, fs, linewidth=4)
    #pylab.xticks([1.5, 2.0, 2.5, 3.0])
    pylab.tight_layout()
    pylab.grid(False)
    pylab.legend()
    pylab.xlabel("$\\ell$")
    pylab.ylabel("$F(\\ell)$")
    pylab.savefig("graphics/fig3_pylab.pdf")
    pylab.show()


def generate_figure_2():
    betas = linspace(1.0, 3.0, 10000)
    deltas = delta(betas)
    Rs = R(betas)
    pylab.figure(figsize=(10, 8))
    pylab.semilogy(betas, deltas, linewidth=4, label="$\\Delta$")
    pylab.semilogy(betas, Rs, linewidth=4, label="$\\mathcal{R}_o$")
    pylab.ylim([0.1, 2.e3])
    pylab.xticks([1.0, 1.5, 2.0, 2.5, 3.0])
    pylab.yticks([0.1, 1.0, 10.0, 100.0, 1000.0])
    pylab.tight_layout()
    pylab.grid(False)
    #pylab.legend()
    #pylab.xlabel("$\\beta$")
    pylab.savefig("graphics/fig2_pylab.pdf")
    pylab.show()

if __name__ == "__main__":

    import sys

    # Generate plot of Delta and R versus beta (Figure 2)
    if False:
        generate_figure_2()

    # To generate contour plots for Q_ref versus \ell_a and \beta_b (Figure 3) 
    # with beta_a as a command line parameter
    if False:
        beta_a = float(sys.argv[1])
        generate_contour_plots(beta_a)

    # To generate curves Q versus \ell_a for different \beta_a, \beta_b (Figure 4) 
    if True:
        generate_tandem_plot()

    # To generate plot of F versus \beta (Equivalent to Figure 5, not used)
    if False:
        generate_plot_of_F()
