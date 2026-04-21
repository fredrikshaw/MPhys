"""
Simple Post-Merger Spin Calculator

Following Buonanno, Kidder, Lehner (2007) - "Estimating the final spin of a binary black hole coalescence"
https://arxiv.org/pdf/0709.3839

This module implements dimensionless formulas for calculating the final spin parameter
of a merged black hole system using the ISCO approximation.
"""

import numpy as np
import matplotlib.pyplot as plt


def calculate_nu(m1, m2):
    """
    Calculate the symmetric mass ratio.
    
    Parameters:
    -----------
    m1 : float
        Mass of first black hole
    m2 : float
        Mass of second black hole
        
    Returns:
    --------
    float
        Symmetric mass ratio nu = m1*m2/(m1+m2)^2
    """
    M = m1 + m2
    return (m1 * m2) / (M**2)


def calculate_Z1(a_star):
    """
    Calculate Z1 parameter for ISCO radius calculation.
    
    Z1 = 1 + (1 - a_*^2)^{1/3} * [(1 + a_*)^{1/3} + (1 - a_*)^{1/3}]
    
    Parameters:
    -----------
    a_star : float
        Dimensionless spin parameter (a_* = a_f/M), must be in range [-1, 1]
        
    Returns:
    --------
    float
        Z1 parameter
    """
    if not -1 <= a_star <= 1:
        raise ValueError(f"a_star must be in range [-1, 1], got {a_star}")
    
    term1 = (1 - a_star**2)**(1/3)
    term2 = (1 + a_star)**(1/3) + (1 - a_star)**(1/3)
    
    return 1 + term1 * term2


def calculate_Z2(Z1, a_star):
    """
    Calculate Z2 parameter for ISCO radius calculation.
    
    Z2 = sqrt(3*a_*^2 + Z1^2)
    
    Parameters:
    -----------
    Z1 : float
        Z1 parameter (can be calculated using calculate_Z1)
    a_star : float
        Dimensionless spin parameter
        
    Returns:
    --------
    float
        Z2 parameter
    """
    return np.sqrt(3 * a_star**2 + Z1**2)


def calculate_r_tilde_isco(a_star, prograde=True):
    """
    Calculate the dimensionless ISCO radius.
    
    r_tilde_ISCO = 3 + Z2 ∓ sqrt[(3-Z1)(3+Z1+2*Z2)]
    
    Where:
    - Upper sign (-) is for prograde orbits (co-rotating)
    - Lower sign (+) is for retrograde orbits (counter-rotating)
    
    Parameters:
    -----------
    a_star : float
        Dimensionless spin parameter of the black hole
    prograde : bool, optional
        True for prograde (co-rotating) orbits, False for retrograde
        Default is True
        
    Returns:
    --------
    float
        Dimensionless ISCO radius r_tilde = r/M
    """
    Z1 = calculate_Z1(a_star)
    Z2 = calculate_Z2(Z1, a_star)
    
    sqrt_term = np.sqrt((3 - Z1) * (3 + Z1 + 2*Z2))
    
    if prograde:
        # Prograde: use minus sign
        r_tilde = 3 + Z2 - sqrt_term
    else:
        # Retrograde: use plus sign
        r_tilde = 3 + Z2 + sqrt_term
    
    return r_tilde


def calculate_L_orb_over_M2(r_tilde, a_star, nu, prograde=True):
    """
    Calculate the dimensionless orbital angular momentum L_orb/M^2.
    
    L_orb/M^2 = nu * [r_tilde^2 ∓ 2*a_star*sqrt(r_tilde) + a_star^2] / 
                     [r_tilde^{3/4} * sqrt(r_tilde^{3/2} - 3*sqrt(r_tilde) ± 2*a_star)]
    
    Where:
    - Upper sign (∓ and ±) corresponds to prograde orbits
    - Lower sign corresponds to retrograde orbits
    
    Parameters:
    -----------
    r_tilde : float
        Dimensionless radius r/M
    a_star : float
        Dimensionless spin parameter
    nu : float
        Symmetric mass ratio
    prograde : bool, optional
        True for prograde orbits, False for retrograde. Default is True
        
    Returns:
    --------
    float
        Dimensionless orbital angular momentum L_orb/M^2
    """
    sqrt_r = np.sqrt(r_tilde)
    
    if prograde:
        # Prograde: ∓ becomes -, ± becomes +
        numerator = r_tilde**2 - 2*a_star*sqrt_r + a_star**2
        denominator_term = r_tilde**(3/2) - 3*sqrt_r + 2*a_star
    else:
        # Retrograde: ∓ becomes +, ± becomes -
        numerator = r_tilde**2 + 2*a_star*sqrt_r + a_star**2
        denominator_term = r_tilde**(3/2) - 3*sqrt_r - 2*a_star
    
    denominator = r_tilde**(3/4) * np.sqrt(denominator_term)
    
    return nu * numerator / denominator


def calculate_L_orb_at_isco(a_star, nu, prograde=True):
    """
    Calculate L_orb/M^2 evaluated at the ISCO radius.
    
    This is a convenience function that combines r_tilde_isco calculation
    with the orbital angular momentum calculation.
    
    Parameters:
    -----------
    a_star : float
        Dimensionless spin parameter
    nu : float
        Symmetric mass ratio
    prograde : bool, optional
        True for prograde orbits, False for retrograde. Default is True
        
    Returns:
    --------
    float
        Dimensionless orbital angular momentum at ISCO
    """
    r_tilde_isco = calculate_r_tilde_isco(a_star, prograde=prograde)
    return calculate_L_orb_over_M2(r_tilde_isco, a_star, nu, prograde=prograde)


def calculate_a_star_final_zero_spin(nu, a_star_guess, prograde=True):
    """
    Calculate the final dimensionless spin for initially non-spinning black holes.
    
    For zero initial spins:
    a_star_f = L_orb(r_ISCO)/M^2
    
    Since L_orb depends on a_star_f, this requires using an initial guess.
    The user should iterate or use self-consistent calculations if needed.
    
    Parameters:
    -----------
    nu : float
        Symmetric mass ratio
    a_star_guess : float
        Initial guess for the final spin parameter (used to calculate r_ISCO and L_orb)
    prograde : bool, optional
        True for prograde orbits, False for retrograde. Default is True
        
    Returns:
    --------
    float
        Final dimensionless spin parameter
    """
    L_orb_M2 = calculate_L_orb_at_isco(a_star_guess, nu, prograde=prograde)
    return L_orb_M2


def calculate_a_star_final(a_star_1, a_star_2, m1, m2, a_star_guess, prograde=True):
    """
    Calculate the final dimensionless spin including initial spins.
    
    a_star_f = L_orb(r_ISCO)/M^2 + 
               a_star_1 * (1/4)(1 + sqrt(1-4*nu))^2 + 
               a_star_2 * (1/4)(1 - sqrt(1-4*nu))^2
    
    Parameters:
    -----------
    a_star_1 : float
        Initial dimensionless spin of first black hole
    a_star_2 : float
        Initial dimensionless spin of second black hole
    m1 : float
        Mass of first black hole
    m2 : float
        Mass of second black hole
    a_star_guess : float
        Initial guess for the final spin (used in L_orb calculation)
    prograde : bool, optional
        True for prograde orbits, False for retrograde. Default is True
        
    Returns:
    --------
    float
        Final dimensionless spin parameter
    """
    nu = calculate_nu(m1, m2)
    
    # Calculate mass fraction coefficients
    sqrt_term = np.sqrt(1 - 4*nu)
    coeff_1 = 0.25 * (1 + sqrt_term)**2
    coeff_2 = 0.25 * (1 - sqrt_term)**2
    
    # Calculate orbital angular momentum contribution
    L_orb_M2 = calculate_L_orb_at_isco(a_star_guess, nu, prograde=prograde)
    
    # Final spin
    a_star_f = L_orb_M2 + a_star_1 * coeff_1 + a_star_2 * coeff_2
    
    return a_star_f


def solve_a_star_self_consistent(nu, a_star_initial=0.0, prograde=True, 
                                  tol=1e-6, max_iter=100):
    """
    Solve for the final spin self-consistently using fixed-point iteration.
    
    Since a_star_f appears on both sides of the equation through L_orb(r_ISCO(a_star_f)),
    we need to iterate to find the self-consistent solution.
    
    Parameters:
    -----------
    nu : float
        Symmetric mass ratio
    a_star_initial : float, optional
        Initial guess for iteration. Default is 0.0
    prograde : bool, optional
        True for prograde orbits, False for retrograde. Default is True
    tol : float, optional
        Convergence tolerance. Default is 1e-6
    max_iter : int, optional
        Maximum number of iterations. Default is 100
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'a_star': Final spin parameter
        - 'r_tilde_isco': ISCO radius at final spin
        - 'L_orb_M2': Orbital angular momentum at ISCO
        - 'converged': Whether iteration converged
        - 'iterations': Number of iterations performed
    """
    a_star = a_star_initial
    
    for i in range(max_iter):
        a_star_new = calculate_a_star_final_zero_spin(nu, a_star, prograde=prograde)
        
        if abs(a_star_new - a_star) < tol:
            r_tilde = calculate_r_tilde_isco(a_star_new, prograde=prograde)
            L_orb = calculate_L_orb_at_isco(a_star_new, nu, prograde=prograde)
            
            return {
                'a_star': a_star_new,
                'r_tilde_isco': r_tilde,
                'L_orb_M2': L_orb,
                'converged': True,
                'iterations': i + 1
            }
        
        a_star = a_star_new
    
    # Did not converge
    return {
        'a_star': a_star,
        'r_tilde_isco': calculate_r_tilde_isco(a_star, prograde=prograde),
        'L_orb_M2': calculate_L_orb_at_isco(a_star, nu, prograde=prograde),
        'converged': False,
        'iterations': max_iter
    }


def plot_final_spin_vs_nu(nu_min=0.01, nu_max=0.25, num_points=100, 
                          prograde=True, save_fig=False, filename='final_spin_vs_nu.png',
                          highlight_mass_ratios=None):
    """
    Plot the final spin parameter as a function of symmetric mass ratio nu
    for initially non-spinning black holes.
    
    Parameters:
    -----------
    nu_min : float, optional
        Minimum value of nu to plot. Default is 0.01
    nu_max : float, optional
        Maximum value of nu to plot. Default is 0.25 (equal mass case)
    num_points : int, optional
        Number of points to compute. Default is 100
    prograde : bool, optional
        True for prograde orbits, False for retrograde. Default is True
    save_fig : bool, optional
        Whether to save the figure. Default is False
    filename : str, optional
        Filename to save the figure. Default is 'final_spin_vs_nu.png'
    highlight_mass_ratios : list of lists, optional
        List of [m1, m2] pairs to highlight on the plot. 
        Example: [[1, 1], [3, 1]] for equal mass and 3:1 ratio
        If None, defaults to [[1, 1], [3, 1]]
        
    Returns:
    --------
    tuple
        (nu_array, a_star_array) - Arrays of nu values and corresponding final spins
    """
    
    # Default mass ratios to highlight
    if highlight_mass_ratios is None:
        highlight_mass_ratios = [[1, 1], [3, 1]]
    nu_values = np.linspace(nu_min, nu_max, num_points)
    a_star_values = []
    r_tilde_isco_values = []
    
    print(f"Computing final spin for {num_points} values of nu...")
    for nu in nu_values:
        result = solve_a_star_self_consistent(nu, prograde=prograde)
        a_star_values.append(result['a_star'])
        r_tilde_isco_values.append(result['r_tilde_isco'])
        
        if not result['converged']:
            print(f"Warning: Did not converge for nu={nu:.4f}")
    
    a_star_values = np.array(a_star_values)
    r_tilde_isco_values = np.array(r_tilde_isco_values)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Final spin vs nu
    ax1.plot(nu_values, a_star_values, 'b-', linewidth=2)
    ax1.set_xlabel(r'Symmetric Mass Ratio $\nu = m_1 m_2 / M^2$', fontsize=12)
    ax1.set_ylabel(r'Final Spin Parameter $a_*^{\mathrm{final}}$', fontsize=12)
    ax1.set_title('Final Spin vs Mass Ratio (Zero Initial Spin)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(nu_min, nu_max)
    
    # Mark highlighted mass ratio cases
    colors = ['ro', 'go', 'mo', 'co', 'yo', 'ko']  # Cycle through colors
    for i, (m1, m2) in enumerate(highlight_mass_ratios):
        nu_highlight = calculate_nu(m1, m2)
        
        # Find closest index in computed values
        idx = np.argmin(np.abs(nu_values - nu_highlight))
        
        # Determine label
        if m1 == m2:
            label = f'Equal mass ({m1}:{m2}): $a_*$ = {a_star_values[idx]:.4f}'
        else:
            # Calculate the ratio for display
            ratio_str = f'{int(m1)}:{int(m2)}' if m1 % 1 == 0 and m2 % 1 == 0 else f'{m1:.1f}:{m2:.1f}'
            label = f'{ratio_str} mass ratio: $a_*$ = {a_star_values[idx]:.4f}'
        
        color = colors[i % len(colors)]
        ax1.plot(nu_values[idx], a_star_values[idx], color, markersize=10, label=label)
    
    ax1.legend(fontsize=10)
    
    # Plot 2: ISCO radius vs nu
    ax2.plot(nu_values, r_tilde_isco_values, 'r-', linewidth=2)
    ax2.set_xlabel(r'Symmetric Mass Ratio $\nu = m_1 m_2 / M^2$', fontsize=12)
    ax2.set_ylabel(r'ISCO Radius $\tilde{r}_{\mathrm{ISCO}}$ (in units of $M$)', fontsize=12)
    ax2.set_title('ISCO Radius at Final Spin', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(nu_min, nu_max)
    
    # Mark highlighted mass ratio cases on ISCO plot
    for i, (m1, m2) in enumerate(highlight_mass_ratios):
        nu_highlight = calculate_nu(m1, m2)
        idx = np.argmin(np.abs(nu_values - nu_highlight))
        
        # Determine label
        if m1 == m2:
            label = f'Equal mass ({m1}:{m2}): $\\tilde{{r}}$ = {r_tilde_isco_values[idx]:.4f}'
        else:
            ratio_str = f'{int(m1)}:{int(m2)}' if m1 % 1 == 0 and m2 % 1 == 0 else f'{m1:.1f}:{m2:.1f}'
            label = f'{ratio_str} mass ratio: $\\tilde{{r}}$ = {r_tilde_isco_values[idx]:.4f}'
        
        color = colors[i % len(colors)]
        ax2.plot(nu_values[idx], r_tilde_isco_values[idx], color, markersize=10, label=label)
    
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")
    
    plt.show()
    
    return nu_values, a_star_values


# Example usage
if __name__ == "__main__":
    print("Post-Merger Spin Calculator")
    print("=" * 50)
    
    # Example 1: Equal mass, non-spinning binary
    print("\nExample 1: Equal mass, non-spinning binary")
    m1, m2 = 1.0, 1.0
    nu = calculate_nu(m1, m2)
    print(f"Mass ratio: m1={m1}, m2={m2}")
    print(f"Symmetric mass ratio nu = {nu:.4f}")
    
    result = solve_a_star_self_consistent(nu, prograde=True)
    print(f"\nSelf-consistent solution:")
    print(f"  Final spin a_star = {result['a_star']:.6f}")
    print(f"  ISCO radius r_tilde = {result['r_tilde_isco']:.6f}")
    print(f"  L_orb/M^2 = {result['L_orb_M2']:.6f}")
    print(f"  Converged: {result['converged']} in {result['iterations']} iterations")
    
    # Example 2: Unequal mass binary
    print("\n" + "=" * 50)
    print("Example 2: Unequal mass binary (3:1 mass ratio)")
    m1, m2 = 3.0, 1.0
    nu = calculate_nu(m1, m2)
    print(f"Mass ratio: m1={m1}, m2={m2}")
    print(f"Symmetric mass ratio nu = {nu:.4f}")
    
    result = solve_a_star_self_consistent(nu, prograde=True)
    print(f"\nSelf-consistent solution:")
    print(f"  Final spin a_star = {result['a_star']:.6f}")
    print(f"  ISCO radius r_tilde = {result['r_tilde_isco']:.6f}")
    print(f"  L_orb/M^2 = {result['L_orb_M2']:.6f}")
    print(f"  Converged: {result['converged']} in {result['iterations']} iterations")
    
    # Example 3: With initial spins
    print("\n" + "=" * 50)
    print("Example 3: Equal mass with initial spins")
    m1, m2 = 1.0, 1.0
    a_star_1, a_star_2 = 0.5, 0.3
    nu = calculate_nu(m1, m2)
    
    # First get self-consistent zero-spin result as guess
    result_zero = solve_a_star_self_consistent(nu, prograde=True)
    
    # Then calculate with initial spins
    a_star_f = calculate_a_star_final(a_star_1, a_star_2, m1, m2, 
                                      result_zero['a_star'], prograde=True)
    
    print(f"Initial spins: a_star_1={a_star_1}, a_star_2={a_star_2}")
    print(f"Final spin a_star_f = {a_star_f:.6f}")
    
    # Example 4: Plot final spin vs nu
    print("\n" + "=" * 50)
    print("Example 4: Plotting final spin vs symmetric mass ratio")
    print("=" * 50)
    
    # Define mass ratios to highlight
    mass_ratios_to_highlight = [[1, 1], [3, 1], [10, 1], [100, 1]]
    print(f"Highlighting mass ratios: {mass_ratios_to_highlight}")
    
    plot_final_spin_vs_nu(nu_min=0.01, nu_max=0.25, num_points=50, 
                         prograde=True, save_fig=True,
                         highlight_mass_ratios=mass_ratios_to_highlight)
