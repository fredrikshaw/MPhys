import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import constants

SOLAR_MASS = 1.988e30 # [kg]

def calc_clmn(l: int, m, n, a, r_g, mu_a):
    num_1 = (2**(4 * l + 4)) * math.factorial(2 * l + n + 1) # factor incorrect in exponent
    denom_1 = (l + n + 1)**(2 * l + 4) * math.factorial(n)
    
    first = num_1 / denom_1
    
    num_2 = math.factorial(l)
    denom_2 = math.factorial(2 * l) * math.factorial(2 * l + 1)
    
    second = (num_2 / denom_2) ** 2
    
    ### product term ###
    prod = 1.0 
    r_plus = calc_r_plus(r_g, a)
    w_plus = calc_w_plus(r_g, a)
    
    for j in range(1, l + 1):
        term = (j**2) * (1 - a**2/r_g**2) + 4 * r_plus**2 * (m * w_plus - mu_a)**2
        prod *= term
    
    return first * second * prod

def calc_r_plus(r_g, a):
    return r_g + np.sqrt(r_g**2 - a**2)

def calc_w_plus(r_g, a):
    first = 1 / (2 * r_g)
    
    num = a / r_g
    denom = 1 + np.sqrt(1 - (a/r_g)**2)
    
    second = num / denom
    return first * second

def calc_alpha(mu_a, r_g):
    return mu_a * r_g

def Gamma(l, m, n, a, r_g, mu_a):
    alpha = calc_alpha(mu_a, r_g)
    r_plus = calc_r_plus(r_g, a)
    w_plus = calc_w_plus(r_g, a)
    
    C_lmn = calc_clmn(l, m, n, a, r_g, mu_a)
    
    # Superradiance rate formula
    return 2 * mu_a * alpha**(4 * l + 4) * r_plus * (m * w_plus - mu_a) * C_lmn

def plot_inverse_superradiance_rate_overlay(blackholemass: float):
    """
    float blackholemass: The mass of the BH in solar masses
    
    """
    ## Set up values to plot ## 
    l_values = [1, 2, 3, 4, 5] # [Dimensionless]
    spins = [0.90, 0.99, 0.999] # [Dimensionless]
    alpha_vals = np.logspace(-2, 1, 500)  # [Dimensionless]

    
    ## Mass[eV] = Mass[SM] x SolarMass[kg] x c^2 
    m_bh_J = blackholemass * SOLAR_MASS * (constants.c) ** 2 # [J]
    m_bh_ev = m_bh_J / constants.e # [eV]
    G_N = 6.708e-57 # [eV^-2]
    r_g = G_N * m_bh_ev # [eV]^-1
    ### CHECK VALUES ###
    print(f"Blackhole Mass: {m_bh_ev:.3e} eV")
    print(f"Graviational radius: {r_g} eV^-1")

    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = ['blue', 'orange', 'red', 'green', 'purple']
    linestyles = ["--", "-.", "-"]
    
    for spin_idx, a_star in enumerate(spins):
        a = a_star * r_g
        
        for l_idx, l in enumerate(l_values):
            m = l
            n = l + 1
            
            gamma_vals = []
            gamma_years = []
            valid_alpha = []
            mu_vals = []
            gamma_rg = []
            
            for alpha in alpha_vals:
                mu_a = alpha / r_g # [eV]
                omega_plus = calc_w_plus(r_g, a)
                
                # Check superradiance condition
                if m * omega_plus > mu_a:
                    try:
                        # Gamma is in units of [eV] (rate in natural units)
                        gamma = Gamma(l, m, n, a, r_g, mu_a) # [eV]
                        gamma_inv = 1 / gamma # [eV]^-1
                        if gamma > 0 and np.isfinite(gamma):
                            gamma_vals.append(gamma)
                            gamma_rg.append(gamma * r_g)
                            gamma_years.append(gamma_inv * 2.086e-23)  # Convert eV^-1 to years
                            valid_alpha.append(alpha)
                            mu_vals.append(mu_a)
                    except (OverflowError, ValueError, ZeroDivisionError):
                        pass
            
            # Plot both spin values on the same axes
            # Use different linestyles for different spins, colors for different l
            ax.semilogy(mu_vals, np.array(gamma_years), 
                     color=colors[l_idx % len(colors)],
                     linestyle=linestyles[spin_idx % len(linestyles)],
                     linewidth=2)
    
    # Format the plot for Gamma^{-1} with inverted y-axis
    ax.set_xlabel(r'$\mu_a$ Axion mass [eV]', fontsize=14)
    # ax.set_ylabel(r'$\Gamma_{lmn} r_g$', fontsize=14)
    ax.set_ylabel(r'$\Gamma_{lmn}^{-1} $ [years]', fontsize=14)
    ax.set_title(r'Superradiance Timescales', fontsize=16)
    color_handles = [Line2D([0], [0], color=colors[i % len(colors)], lw=4, linestyle='-') for i, _ in enumerate(l_values)]
    color_labels = [fr"$\ell$={l}" for l in l_values]
    legend1 = ax.legend(color_handles, color_labels, title=r'orbital ($\ell$)', fontsize=10, loc='upper right', frameon=True)

    # create a linestyle legend (separate box) for spins
    style_handles = [Line2D([0], [0], color='black', lw=2, linestyle=linestyles[i % len(linestyles)]) for i, _ in enumerate(spins)]
    style_labels = [f"a*={a_star:.3f}" for a_star in spins]
    legend2 = ax.legend(style_handles, style_labels, title=r'spin ($a^*$)', fontsize=10, loc='upper left', frameon=True)

    ax.add_artist(legend1)
    ax.add_artist(legend2)

    ax.grid(True, which="both", ls="-", alpha=0.2)
    # ax.set_xlim(0, 0.5)
    ax.set_ylim(1e2, 1e-5)
    
    plt.tight_layout()
    plt.savefig("SRRateOutput.pdf")
    plt.show()

def main():
    blackholemass = 1e1 # solar mass(es) [Dimensionless]

    plot_inverse_superradiance_rate_overlay(blackholemass)

if __name__ == "__main__":
    main()