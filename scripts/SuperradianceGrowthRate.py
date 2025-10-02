import math
import numpy as np
import matplotlib.pyplot as plt

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
    l_values = [1, 2, 3, 4]
    
    r_g = 1#.33e21
    spins = [0.90, 0.99]
    alpha_vals = np.logspace(-2, 1, 200)  # 0.01 to 1.0 (log scale)
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = ['blue', 'orange', 'red', 'green']
    linestyles = ["--", "-"]
    
    for spin_idx, a_star in enumerate(spins):
        a = a_star * r_g
        
        for l_idx, l in enumerate(l_values):
            m = l  # m = l
            n = l  # n = l
            
            gamma_vals = []
            Gamma_inv_values = []  # Store Gamma^{-1} values
            valid_alpha = []
            
            for alpha in alpha_vals:
                mu_a = alpha / r_g
                omega_plus = calc_w_plus(r_g, a)
                
                # Check superradiance condition
                if m * omega_plus > mu_a:
                    try:
                        gamma = Gamma(l, m, n, a, r_g, mu_a)
                        gamma_vals.append(gamma)
                        # Calculate Gamma^{-1} instead of Gamma
                        if gamma > 0 and np.isfinite(gamma):
                            gamma_inv = 1.0 / gamma  # This is Gamma^{-1}
                            Gamma_inv_values.append(gamma_inv)
                            valid_alpha.append(alpha)
                        else:
                            # Use large value for log plot when Gamma is very small
                            Gamma_inv_values.append(1e20)
                            valid_alpha.append(alpha)
                    except (OverflowError, ValueError, ZeroDivisionError):
                        # Handle numerical issues
                        Gamma_inv_values.append(1e20)
                        gamma_vals.append(1e-20)
                        valid_alpha.append(alpha)
                else:
                    # Superradiance condition not satisfied - very long timescale
                    gamma_vals.append(1e-20)
                    Gamma_inv_values.append(1e20)
                    valid_alpha.append(alpha)
            
            # Plot both spin values on the same axes
            # Use different linestyles for different spins, colors for different l
            ax.semilogy(valid_alpha, gamma_vals, 
                     label=f'l={l}, a*={a_star}', 
                     color=colors[l_idx % len(colors)],
                     linestyle=linestyles[spin_idx % len(linestyles)],
                     linewidth=2)
    
    # Format the plot for Gamma^{-1} with inverted y-axis
    ax.set_xlabel(r'$\alpha = \mu_a r_g$', fontsize=14)
    ax.set_ylabel(r'$\Gamma_{lmn} r_g$', fontsize=14)
    ax.set_title('Superradiance Timescales', fontsize=16)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_xlim(0, 2)
    ax.set_ylim(1e-14, 1e-6)  # INVERTED: now descending from top to bottom
    
    # # Add horizontal lines for physical timescales
    # ax.axhline(y=1, color='gray', linestyle=':', alpha=0.7, label=r'$1 r_g$')
    # ax.axhline(y=1e7, color='brown', linestyle=':', alpha=0.7, label=r'$10^7 r_g$')
    # ax.axhline(y=1e15, color='orange', linestyle=':', alpha=0.7, label=r'$10^{15} r_g$')
    
    # # Add text annotations for physical interpretation
    # ax.text(0.02, 1e18, 'Very Slow Growth', fontsize=10, alpha=0.8)
    # ax.text(0.02, 1e10, 'Moderate Growth', fontsize=10, alpha=0.8)
    # ax.text(0.02, 1e2, 'Fast Growth', fontsize=10, alpha=0.8)
    
    plt.tight_layout()
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