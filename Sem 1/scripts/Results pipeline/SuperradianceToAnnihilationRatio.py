from ParamCalculator import (
    calc_rg_from_bh_mass,
    calc_omega_ann,
    calc_annihilation_rate,
    calc_superradiance_rate,
    calc_n_max,
    calc_delta_astar,
    G_N
)
import numpy as np


def calculate_sr_to_ann_ratio(bh_mass_solar, alpha, astar_init, n, m):
    """
    Calculate the ratio: Gamma_SR / (n_max * Gamma_ann)
    
    Args:
        bh_mass_solar (float): Black hole mass [M_☉]
        alpha (float): Fine structure constant (dimensionless)
        astar_init (float): Initial dimensionless spin parameter (dimensionless)
        n (int): Principal quantum number (dimensionless)
        m (int): Azimuthal quantum number (dimensionless, typically m=l for superradiance)
    
    Returns:
        dict: Dictionary containing:
            - 'ratio': Gamma_SR / (n_max * Gamma_ann) (dimensionless)
            - 'sr_rate': Superradiance rate [eV]
            - 'ann_rate': Annihilation rate [eV]
            - 'n_max': Maximum occupation number (dimensionless)
            - 'level': Spectroscopic notation (e.g., '2p', '3d')
    """
    # Calculate gravitational radius
    r_g = calc_rg_from_bh_mass(bh_mass_solar)
    
    # For superradiance, l = m
    l = m
    
    # Get spectroscopic notation
    orbital_letters = {1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h', 6: 'i', 7: 'j', 8: 'k'}
    if l not in orbital_letters:
        raise ValueError(f"l={l} not supported. Use l in range 1-8.")
    level = f"{n}{orbital_letters[l]}"
    
    # Calculate annihilation frequency and rate
    omega_ann = calc_omega_ann(r_g, alpha, n)
    ann_rate = calc_annihilation_rate(level, alpha, omega_ann, G_N=G_N, r_g=r_g)
    
    # Calculate superradiance rate
    sr_rate = calc_superradiance_rate(l, m, n, astar_init, r_g, alpha)
    
    # Calculate n_max
    bh_mass_ev = r_g / G_N
    delta_astar = calc_delta_astar(astar_init, r_g, alpha, n, m)
    n_max = calc_n_max(bh_mass_ev, delta_astar, m)
    
    # Calculate ratio
    ratio = sr_rate / (n_max * ann_rate)
    
    return {
        'ratio': ratio,
        'sr_rate': sr_rate,
        'ann_rate': ann_rate,
        'n_max': n_max,
        'level': level
    }


def print_ratio_table(bh_mass_solar, alpha, astar_init, n_range, m_values):
    """
    Print a table of Gamma_SR / (n_max * Gamma_ann) ratios for different states.
    
    Args:
        bh_mass_solar (float): Black hole mass [M_☉]
        alpha (float): Fine structure constant (dimensionless)
        astar_init (float): Initial dimensionless spin parameter (dimensionless)
        n_range (list or range): Range of principal quantum numbers
        m_values (list): List of m (=l) values to calculate
    """
    print(f"\nRatio: Gamma_SR / (n_max * Gamma_ann)")
    print(f"Black hole mass: {bh_mass_solar:.2e} M_☉")
    print(f"Alpha: {alpha}")
    print(f"Initial spin: {astar_init}")
    print("\n" + "="*80)
    
    # Header
    orbital_letters = {1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h', 6: 'i', 7: 'j', 8: 'k'}
    header = "n \\ m"
    for m in m_values:
        if m in orbital_letters:
            header += f" | {orbital_letters[m]:>10s}"
    print(header)
    print("-"*80)
    
    # Calculate and print ratios
    for n in n_range:
        row = f"{n:2d}   "
        for m in m_values:
            if m >= n:  # Skip physically invalid states (m must be < n)
                row += f" | {'---':>10s}"
                continue
            
            try:
                result = calculate_sr_to_ann_ratio(bh_mass_solar, alpha, astar_init, n, m)
                ratio = result['ratio']
                row += f" | {ratio:>10.3e}"
            except Exception as e:
                row += f" | {'ERROR':>10s}"
        print(row)
    print("="*80)


def print_detailed_results(bh_mass_solar, alpha, astar_init, n, m):
    """
    Print detailed results for a specific state.
    
    Args:
        bh_mass_solar (float): Black hole mass [M_☉]
        alpha (float): Fine structure constant (dimensionless)
        astar_init (float): Initial dimensionless spin parameter (dimensionless)
        n (int): Principal quantum number (dimensionless)
        m (int): Azimuthal quantum number (dimensionless)
    """
    result = calculate_sr_to_ann_ratio(bh_mass_solar, alpha, astar_init, n, m)
    
    print(f"\nDetailed results for level {result['level']} (n={n}, l=m={m})")
    print(f"Black hole mass: {bh_mass_solar:.2e} M_☉")
    print(f"Alpha: {alpha}")
    print(f"Initial spin: {astar_init}")
    print("-"*60)
    print(f"Superradiance rate:     Gamma_SR   = {result['sr_rate']:.6e} eV")
    print(f"Annihilation rate:      Gamma_ann  = {result['ann_rate']:.6e} eV")
    print(f"Maximum occupation:     n_max      = {result['n_max']:.6e}")
    print(f"Product:                n_max * Gamma_ann = {result['n_max'] * result['ann_rate']:.6e} eV")
    print(f"\nRatio:                  Gamma_SR / (n_max * Gamma_ann) = {result['ratio']:.6e}")
    print("="*60)


if __name__ == "__main__":
    # Example parameters
    bh_mass_solar = 1e-11  # M_☉
    alpha = 0.1
    astar_init = 0.687
    
    # Print table for various states
    n_range = range(2, 11)  # n = 2 to 10
    m_values = [1, 2, 3, 4, 5, 6, 7, 8]  # p, d, f, g, h, i, j, k
    
    print_ratio_table(bh_mass_solar, alpha, astar_init, n_range, m_values)
    
    # Print detailed results for a specific state (e.g., 2p)
    print("\n")
    print_detailed_results(bh_mass_solar, alpha, astar_init, n=2, m=1)
