import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def calc_clmn(l: int, m, n, a, r_g, mu_a):
    # Use ** for exponentiation
    num_1 = (2**(4 * l + 2)) * math.factorial(2 * l + n + 1)
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

def plot_inverse_superradiance_rate_overlay():
    l_values = [1, 2, 3, 4, 5]
    
    m_bh_sm = 1e1
    m_bh_ev = m_bh_sm * 1.116e66  # 10^-11 solar masses in kg
    G_N = 6.708e-57  # m^3 kg^-1 s^-2
    r_g = G_N * m_bh_ev
    print(r_g)
    spins = [0.90, 0.99, 0.999]
    alpha_vals = np.logspace(-2, 1, 200)  # 0.01 to 1.0 (log scale)
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = ['blue', 'orange', 'red', 'green', 'purple']
    linestyles = ["--", "-.", "-"]
    
    for spin_idx, a_star in enumerate(spins):
        a = a_star * r_g
        
        for l_idx, l in enumerate(l_values):
            m = l  # m = l
            n = l  # n = l
            
            gamma_vals = []
            gamma_years = []
            valid_alpha = []
            velocity_vals = []
            mu_vals = []
            
            for alpha in alpha_vals:
                mu_a = alpha / r_g
                omega_plus = calc_w_plus(r_g, a)
                
                # Check superradiance condition
                if m * omega_plus > mu_a:
                    try:
                        gamma = Gamma(l, m, n, a, r_g, mu_a)
                        # Calculate Gamma^{-1} instead of Gamma
                        if gamma > 0 and np.isfinite(gamma):
                            gamma_vals.append(gamma)
                            gamma_years.append(gamma * 6.58e-16 * 3.154e7)  # Convert eV^-1 to years
                            valid_alpha.append(alpha)
                            velocity_vals.append(alpha / l)  # v = alpha / l
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
    ax.set_ylabel(r'$\Gamma_{lmn} [years^{-1}]$', fontsize=14)
    ax.set_title(r'Superradiance Timescales', fontsize=16)
    color_handles = [Line2D([0], [0], color=colors[i % len(colors)], lw=4, linestyle='-') for i, _ in enumerate(l_values)]
    color_labels = [f"l={l}" for l in l_values]
    legend1 = ax.legend(color_handles, color_labels, title='orbital (l)', fontsize=10, loc='upper right', frameon=True)

    # create a linestyle legend (separate box) for spins
    style_handles = [Line2D([0], [0], color='black', lw=2, linestyle=linestyles[i % len(linestyles)]) for i, _ in enumerate(spins)]
    style_labels = [f"a*={a_star:.3f}" for a_star in spins]
    legend2 = ax.legend(style_handles, style_labels, title='spin (a*)', fontsize=10, loc='upper left', frameon=True)

    ax.add_artist(legend1)
    ax.add_artist(legend2)

    ax.grid(True, which="both", ls="-", alpha=0.2)
    # ax.set_xlim(0, 0.5)
    # ax.set_ylim(1e-16, 1e-6)
    
    plt.tight_layout()
    plt.savefig("SRRateOutput.pdf")
    plt.show()

def demonstrate_plot_features():
    """Explain what the plot shows"""
    print("Plot Features:")
    print("=" * 50)
    print("1. OVERLAY: Both spin values (a* = 0.90 and 0.99) on same plot")
    print("2. COLOR CODING: Different colors for different l values")
    print("   - Blue: l=1, Red: l=2, Green: l=3, Purple: l=4")
    print("3. LINE STYLE: Solid lines: a*=0.99, Dashed lines: a*=0.90")
    print("4. INVERTED Y-AXIS: Timescale DESCENDS from top to bottom")
    print("   - Top: Long timescales (slow growth)")
    print("   - Bottom: Short timescales (fast growth)")
    print("5. PHYSICAL MEANING:")
    print("   - Lower curves = Faster instability growth")
    print("   - Higher spin (a*=0.99) = Faster growth than lower spin (a*=0.90)")
    print("   - Lower l values = Much faster growth than higher l values")
    print("=" * 50)

if __name__ == "__main__":
    demonstrate_plot_features()
    plot_inverse_superradiance_rate_overlay()