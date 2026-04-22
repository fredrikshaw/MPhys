"""
Improved superradiance growth rate calculation with piecewise approximations
Valid across different α/ℓ regimes
"""

import numpy as np
import math
from scipy import constants

SOLAR_MASS = 1.988e30  # [kg]

def calc_r_plus(r_g, a):
    """Calculate r_+ (outer horizon radius)"""
    return r_g + np.sqrt(r_g**2 - a**2)

def calc_w_plus(r_g, a):
    """Calculate angular velocity at horizon ω₊"""
    first = 1 / (2 * r_g)
    num = a / r_g
    denom = 1 + np.sqrt(1 - (a / r_g)**2)
    second = num / denom
    return first * second

def calc_clmn(l, m, n, a, r_g, mu_a):
    """
    Calculate C_lmn coefficient from Arvanitaki & Dubovsky (2011)
    Valid for small α/ℓ regime
    """
    num_1 = (2 ** (4 * l + 4)) * math.factorial(2 * l + n + 1)
    denom_1 = (l + n + 1) ** (2 * l + 4) * math.factorial(n)
    first = num_1 / denom_1

    num_2 = math.factorial(l)
    denom_2 = math.factorial(2 * l) * math.factorial(2 * l + 1)
    second = (num_2 / denom_2) ** 2

    prod = 1.0
    r_plus = calc_r_plus(r_g, a)
    w_plus = calc_w_plus(r_g, a)

    for j in range(1, l + 1):
        term = (j ** 2) * (1 - a**2 / r_g**2) + 4 * r_plus**2 * (m * w_plus - mu_a)**2
        prod *= term

    return first * second * prod

def calc_gamma_small_alpha(l, m, n, a, r_g, mu_a):
    """
    Analytic formula for small α/ℓ ≪ 1 regime
    From Arvanitaki & Dubovsky (2011) Eq. 2.8-2.9
    """
    alpha = mu_a * r_g
    r_plus = calc_r_plus(r_g, a)
    w_plus = calc_w_plus(r_g, a)
    C_lmn = calc_clmn(l, m, n, a, r_g, mu_a)
    
    # Corrected formula with (2μₐr₊) instead of α
    return 2 * mu_a * (2 * mu_a * r_plus) ** (4 * l + 4) * r_plus * (m * w_plus - mu_a) * C_lmn

def calc_gamma_wkb(l, m, n, a, r_g, mu_a):
    """
    WKB approximation for large α/ℓ ~ 0.5 regime
    From paper: Γ ∝ e^(-3.7α) = (0.15)^ℓ
    """
    alpha = mu_a * r_g
    
    # Base normalization from typical rates in the paper
    # For ℓ=1 levels near α~0.3, Γ ~ 1e-7/r_g
    base_rate = 1e-7 / r_g
    
    # WKB scaling
    wkb_factor = np.exp(-3.7 * alpha)
    
    # Additional ℓ dependence from paper
    ell_factor = (0.15) ** l
    
    return base_rate * wkb_factor * ell_factor

def calc_gamma_improved(l, m, n, a, r_g, mu_a, a_star=None, verbose=False):
    """
    Piecewise superradiance rate calculation valid for all α/ℓ
    
    Parameters:
    -----------
    l, m, n : int
        Quantum numbers
    a : float
        Black hole spin parameter [eV⁻¹] (NOT dimensionless a*)
    r_g : float  
        Gravitational radius [eV⁻¹]
    mu_a : float
        Axion mass [eV]
    a_star : float, optional
        Dimensionless spin a* = a/r_g (for convenience)
    verbose : bool, optional
        Print debug information
        
    Returns:
    --------
    gamma : float
        Superradiance rate [eV]
    """
    alpha = mu_a * r_g
    
    if a_star is None:
        a_star = a / r_g
    
    alpha_over_l = alpha / l
    
    # Check superradiance condition
    w_plus = calc_w_plus(r_g, a)
    sr_condition = m * w_plus - mu_a
    
    if verbose:
        print(f"Debug: α = {alpha:.3f}, α/ℓ = {alpha_over_l:.3f}")
        print(f"Debug: mω₊ = {m * w_plus:.3e}, μₐ = {mu_a:.3e}")
        print(f"Debug: Superradiance condition = {sr_condition:.3e}")
    
    if sr_condition <= 0:
        if verbose:
            print("Debug: Superradiance condition not satisfied")
        return 0.0  # No superradiance
    
    # Define regime boundaries
    SMALL_ALPHA_THRESHOLD = 0.2
    LARGE_ALPHA_THRESHOLD = 0.2
    
    if alpha_over_l < SMALL_ALPHA_THRESHOLD:
        # Small α/ℓ regime - use analytic formula
        gamma = calc_gamma_small_alpha(l, m, n, a, r_g, mu_a)
        regime = "small_alpha"
        
    elif alpha_over_l > LARGE_ALPHA_THRESHOLD:
        # Large α/ℓ regime - use WKB approximation  
        gamma = calc_gamma_wkb(l, m, n, a, r_g, mu_a)
        regime = "wkb"
        
    else:
        # Intermediate regime - interpolate between approximations
        gamma_small = calc_gamma_small_alpha(l, m, n, a, r_g, mu_a)
        gamma_large = calc_gamma_wkb(l, m, n, a, r_g, mu_a)
        
        # Linear interpolation in α/ℓ space
        t = (alpha_over_l - SMALL_ALPHA_THRESHOLD) / (LARGE_ALPHA_THRESHOLD - SMALL_ALPHA_THRESHOLD)
        gamma = (1 - t) * gamma_small + t * gamma_large
        regime = "interpolated"
    
    if verbose:
        print(f"Debug: Using {regime} regime, Γ = {gamma:.3e} eV")
    
    # Ensure positive rate (numerical safety)
    return max(gamma, 0.0)

def calc_gamma_improved_with_units(l, m, n, a_star, M_bh_solar, mu_a_ev, verbose=False):
    """
    Convenience function with astronomical units
    
    Parameters:
    -----------
    l, m, n : int
        Quantum numbers
    a_star : float
        Dimensionless spin parameter (0 < a_star < 1)
    M_bh_solar : float
        Black hole mass in solar masses
    mu_a_ev : float
        Axion mass in eV
    verbose : bool, optional
        Print debug information
        
    Returns:
    --------
    gamma_ev : float
        Superradiance rate [eV]
    gamma_yr : float  
        Superradiance rate [yr⁻¹]
    """
    # Constants
    G_N = 6.708e-57  # eV⁻²
    ev_to_yr = 1.52e15  # Conversion: 1 eV = 1.52e15 yr⁻¹
    
    # Convert BH mass to eV
    m_bh_J = M_bh_solar * SOLAR_MASS * constants.c**2  # [J]
    m_bh_ev = m_bh_J / constants.e  # [eV]
    r_g = G_N * m_bh_ev  # [eV⁻¹]
    
    # Calculate spin parameter a from a_star
    a = a_star * r_g  # [eV⁻¹]
    
    if verbose:
        print(f"Debug: M_BH = {M_bh_solar} M⊙")
        print(f"Debug: r_g = {r_g:.3e} eV⁻¹")
        print(f"Debug: a = {a:.3e} eV⁻¹")
    
    # Calculate improved gamma
    gamma_ev = calc_gamma_improved(l, m, n, a, r_g, mu_a_ev, a_star=a_star, verbose=verbose)
    gamma_yr = gamma_ev * ev_to_yr
    
    return gamma_ev, gamma_yr

# Example usage and testing
if __name__ == "__main__":
    # Test case: 5g level from Table 1
    l, m, n = 4, 4, 5  # 5g level
    a_star = 0.9
    M_bh_solar = 10.0
    mu_a_ev = 1.1e-11  # Corresponds to α = 1.2 for 10M⊙ BH
    
    print("Superradiance Rate Calculation")
    print("=" * 50)
    print(f"Level: {n}g (n={n}, l={l}, m={m})")
    print(f"BH: {M_bh_solar} M⊙, a* = {a_star}")
    print(f"μ_a = {mu_a_ev:.2e} eV")
    
    gamma_ev, gamma_yr = calc_gamma_improved_with_units(
        l, m, n, a_star, M_bh_solar, mu_a_ev, verbose=True
    )
    
    print(f"Γ = {gamma_ev:.2e} eV")
    print(f"Γ = {gamma_yr:.2e} yr⁻¹")
    
    if gamma_yr > 0:
        print(f"Γ⁻¹ = {1/gamma_yr:.2e} years")
    else:
        print("Γ⁻¹ = ∞ (no superradiance)")
    
    print("=" * 50)
    
