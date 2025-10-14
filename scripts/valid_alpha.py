"""
Plot superradiance rates with log scale to see negative values near boundary
"""

import numpy as np
import matplotlib.pyplot as plt
from improved_superradiance import calc_gamma_improved_with_units, calc_w_plus

def plot_gamma_with_log_condition():
    """
    Plot superradiance rates and show the superradiance condition on log scale
    """
    # Fixed parameters
    M_bh_solar = 10.0
    a_star = 0.9
    
    # Quantum numbers
    n_g, l_g, m_g = 5, 4, 4  # 5g level
    n_e, l_e, m_e = 6, 4, 4  # 6g level
    
    # Extended alpha range to see where it breaks
    alpha_min, alpha_max = 0.1, 2.0
    alpha_values = np.linspace(alpha_min, alpha_max, 5000)
    
    # Calculate r_g
    G_N = 6.708e-57
    SOLAR_MASS = 1.988e30
    m_bh_J = M_bh_solar * SOLAR_MASS * (3e8)**2
    m_bh_ev = m_bh_J / 1.602e-19
    r_g = G_N * m_bh_ev
    
    print(f"Black hole parameters:")
    print(f"  M = {M_bh_solar} M⊙, a* = {a_star}")
    print(f"  r_g = {r_g:.3e} eV⁻¹")
    
    # Calculate ω₊ for this black hole
    a = a_star * r_g
    w_plus = calc_w_plus(r_g, a)
    print(f"  ω₊ = {w_plus:.3e} eV")
    print(f"  Maximum μₐ for superradiance (m=4): {4*w_plus:.3e} eV")
    print(f"  Corresponding α_max: {4*w_plus*r_g:.3f}")
    print()
    
    # Arrays to store results
    gamma_g_values = []
    gamma_e_values = []
    mu_a_values = []
    valid_alpha = []
    sr_condition_values_g = []  # Store the actual condition value
    sr_condition_values_e = []
    w_eff_g_values = []
    w_eff_e_values = []
    
    # Calculate for each alpha
    for alpha in alpha_values:
        mu_a = alpha / r_g
        
        # Calculate exact energies (including binding energy)
        w_eff_g = mu_a * (1 - alpha**2/(2*n_g**2))  # Energy for 5g level
        w_eff_e = mu_a * (1 - alpha**2/(2*n_e**2))  # Energy for 6g level
        
        # Superradiance condition: mω₊ > ω
        condition_value_g = m_g * w_plus - w_eff_g
        condition_value_e = m_e * w_plus - w_eff_e
        
        sr_condition_values_g.append(condition_value_g)
        sr_condition_values_e.append(condition_value_e)
        w_eff_g_values.append(w_eff_g)
        w_eff_e_values.append(w_eff_e)
        
        # Only calculate rates if condition is satisfied
        if condition_value_g > 0 and condition_value_e > 0:
            gamma_g_ev, gamma_g_yr = calc_gamma_improved_with_units(
                l_g, m_g, n_g, a_star, M_bh_solar, mu_a, verbose=False
            )
            gamma_e_ev, gamma_e_yr = calc_gamma_improved_with_units(
                l_e, m_e, n_e, a_star, M_bh_solar, mu_a, verbose=False
            )
            
            if gamma_g_yr > 0 and gamma_e_yr > 0:
                gamma_g_values.append(gamma_g_yr)
                gamma_e_values.append(gamma_e_yr)
                mu_a_values.append(mu_a)
                valid_alpha.append(alpha)
            else:
                gamma_g_values.append(0)
                gamma_e_values.append(0)
                mu_a_values.append(mu_a)
                valid_alpha.append(alpha)
        else:
            # Store zeros for invalid points
            gamma_g_values.append(0)
            gamma_e_values.append(0)
            mu_a_values.append(mu_a)
            valid_alpha.append(alpha)
    
    # Convert to numpy arrays
    alpha_values = np.array(alpha_values)
    sr_condition_values_g = np.array(sr_condition_values_g)
    sr_condition_values_e = np.array(sr_condition_values_e)
    w_eff_g_values = np.array(w_eff_g_values)
    w_eff_e_values = np.array(w_eff_e_values)
    
    # Create the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Superradiance rates (top left)
    mask_valid = np.array(gamma_g_values) > 0
    if np.any(mask_valid):
        valid_alpha_plot = np.array(valid_alpha)[mask_valid]
        gamma_g_plot = np.array(gamma_g_values)[mask_valid]
        gamma_e_plot = np.array(gamma_e_values)[mask_valid]
        
        ax1.semilogy(valid_alpha_plot, gamma_g_plot, 'b-', linewidth=2, label=r'$\Gamma_g$ (5g)')
        ax1.semilogy(valid_alpha_plot, gamma_e_plot, 'r--', linewidth=2, label=r'$\Gamma_e$ (6g)')
    
    # Mark the theoretical boundary
    boundary_alpha = 4 * w_plus * r_g
    ax1.axvline(x=boundary_alpha, color='red', linestyle=':', alpha=0.7, 
                label=f'Theoretical limit: α = {boundary_alpha:.2f}')
    
    ax1.set_xlabel(r'Gravitational coupling $\alpha$', fontsize=12)
    ax1.set_ylabel(r'Superradiance rate $\Gamma$ [yr$^{-1}$]', fontsize=12)
    ax1.set_title('Superradiance Rates', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xlim(alpha_min, alpha_max)
    
    # Plot 2: Superradiance condition values on LOG scale (top right)
    # Use symlog to show both positive and negative values
    ax2.semilogy(alpha_values, np.abs(sr_condition_values_g), 'b-', linewidth=2, 
                label=r'$|m\omega_+ - \omega_g|$ (5g)')
    ax2.semilogy(alpha_values, np.abs(sr_condition_values_e), 'r--', linewidth=2, 
                label=r'$|m\omega_+ - \omega_e|$ (6g)')
    
    # Add a line at y=0 to mark the boundary
    ax2.axhline(y=1e-30, color='k', linestyle='-', alpha=0.3, label='Zero boundary')
    
    # Mark where condition becomes negative
    zero_crossing_g = alpha_values[np.where(np.diff(np.signbit(sr_condition_values_g)))[0]]
    zero_crossing_e = alpha_values[np.where(np.diff(np.signbit(sr_condition_values_e)))[0]]
    
    for crossing in zero_crossing_g:
        ax2.axvline(x=crossing, color='blue', linestyle=':', alpha=0.7, 
                   label=f'5g boundary: α = {crossing:.3f}')
    for crossing in zero_crossing_e:
        ax2.axvline(x=crossing, color='red', linestyle=':', alpha=0.7, 
                   label=f'6g boundary: α = {crossing:.3f}')
    
    ax2.set_xlabel(r'Gravitational coupling $\alpha$', fontsize=12)
    ax2.set_ylabel(r'$|m\omega_+ - \omega|$ [eV] (Log Scale)', fontsize=12)
    ax2.set_title('Superradiance Condition (Absolute Value, Log Scale)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim(alpha_min, alpha_max)
    
    # Plot 3: Energies (bottom left)
    ax3.plot(alpha_values, np.full_like(alpha_values, m_g * w_plus), 'k-', 
             linewidth=2, label=r'$m\omega_+$')
    ax3.plot(alpha_values, w_eff_g_values, 'b-', linewidth=2, label=r'$\omega_g$ (5g)')
    ax3.plot(alpha_values, w_eff_e_values, 'r--', linewidth=2, label=r'$\omega_e$ (6g)')
    ax3.set_xlabel(r'Gravitational coupling $\alpha$', fontsize=12)
    ax3.set_ylabel('Energy [eV]', fontsize=12)
    ax3.set_title('Energy Comparison', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    ax3.set_xlim(alpha_min, alpha_max)
    
    # Plot 4: Ratio of rates (bottom right)
    if np.any(mask_valid):
        ratio = np.array(gamma_e_values)[mask_valid] / np.array(gamma_g_values)[mask_valid]
        ax4.plot(valid_alpha_plot, ratio, 'g-', linewidth=2)
        ax4.set_xlabel(r'Gravitational coupling $\alpha$', fontsize=12)
        ax4.set_ylabel(r'Ratio $\Gamma_e / \Gamma_g$', fontsize=12)
        ax4.set_title('Ratio of Superradiance Rates', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1, color='k', linestyle=':', alpha=0.7, label=r'$\Gamma_e = \Gamma_g$')
        ax4.legend(fontsize=11)
        ax4.set_xlim(alpha_min, alpha_max)
    
    # Add information
    info_text = (
        f'$M_{{BH}} = {M_bh_solar}\,M_\\odot$\n'
        f'$a^* = {a_star}$\n'
        f'$\\omega_+ = {w_plus:.3e}$ eV\n'
        f'Max $\\alpha$ (theory): {boundary_alpha:.2f}'
    )
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save and show
    filename = "superradiance_log_condition.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{filename}'")
    
    # Print detailed information about the boundary
    print("\nDetailed boundary analysis:")
    print("-" * 50)
    
    # Find where conditions cross zero
    if len(zero_crossing_g) > 0:
        print(f"5g level superradiance stops at α = {zero_crossing_g[0]:.4f}")
    if len(zero_crossing_e) > 0:
        print(f"6g level superradiance stops at α = {zero_crossing_e[0]:.4f}")
    
    # Print condition values near the boundary
    boundary_idx = np.argmin(np.abs(alpha_values - 1.25))
    print(f"\nNear α = 1.25:")
    print(f"  5g condition: {sr_condition_values_g[boundary_idx]:.2e} eV")
    print(f"  6g condition: {sr_condition_values_e[boundary_idx]:.2e} eV")
    print(f"  ω_g = {w_eff_g_values[boundary_idx]:.2e} eV")
    print(f"  ω_e = {w_eff_e_values[boundary_idx]:.2e} eV")
    print(f"  mω₊ = {m_g * w_plus:.2e} eV")
    
    plt.show()

if __name__ == "__main__":
    plot_gamma_with_log_condition()