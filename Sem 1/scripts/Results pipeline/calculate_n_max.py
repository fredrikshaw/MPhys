"""
Calculate maximum occupation number (n_max) for annihilation levels.

This script calculates n_max for all valid annihilation levels using the same
input parameters as StrainVsFrequencyPlot.py.
"""

import numpy as np
from ParamCalculator import (
    calc_rg_from_bh_mass,
    calc_omega_ann,
    calc_delta_astar,
    calc_n_max,
    G_N
)
from ConvertedFunctions import diff_power_ann_dict


def parse_level(level_str):
    """
    Parse level string (e.g., '2p', '3d') into quantum numbers.
    
    Args:
        level_str (str): Level identifier (e.g., '2p', '3d', '4f')
    
    Returns:
        tuple: (n, m) where n is principal quantum number and m is azimuthal
    """
    m_lookup = {
        "s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5,
        "i": 6, "k": 7, "l": 8, "m": 9, "n": 10, "o": 11,
        "q": 12, "r": 13, "t": 14,
    }
    n = int(level_str[0])
    m = m_lookup[level_str[1]]
    return n, m


def calculate_n_max_for_level(level_str, bh_mass_solar, alpha, astar_init):
    """
    Calculate n_max for a specific annihilation level.
    
    Args:
        level_str (str): Level identifier (e.g., '2p', '3d')
        bh_mass_solar (float): Black hole mass [M_☉]
        alpha (float): Fine structure constant (dimensionless)
        astar_init (float): Initial spin parameter (dimensionless)
    
    Returns:
        dict: Dictionary with level info and n_max, or None if invalid
    """
    # Parse level
    n, m = parse_level(level_str)
    
    # Calculate gravitational radius
    r_g = calc_rg_from_bh_mass(bh_mass_solar)  # [eV^-1]
    
    # Calculate black hole mass in eV
    bh_mass_eV = r_g / G_N  # [eV]
    
    # Calculate frequency
    omega = calc_omega_ann(r_g, alpha, n)  # [eV]
    
    # Calculate spin parameter change
    delta_astar = calc_delta_astar(astar_init, r_g, alpha, n, m)
    
    # Check if level is valid (delta_astar should be positive)
    if delta_astar <= 0:
        return None
    
    # Calculate maximum occupation number
    n_max = calc_n_max(bh_mass_eV, delta_astar, m)
    
    return {
        'level': level_str,
        'n': n,
        'l': m,  # For annihilations, l = m
        'm': m,
        'alpha': alpha,
        'omega_eV': omega,
        'bh_mass_eV': bh_mass_eV,
        'delta_astar': delta_astar,
        'n_max': n_max
    }


def main():
    """Main function to calculate n_max for all annihilation levels."""
    # Input parameters (matching StrainVsFrequencyPlot.py)
    bh_mass_solar = 1e-6  # Black hole mass in solar masses
    alpha = 0.1          # Fine structure constant
    astar_init = 0.687   # Initial spin parameter
    
    print("=" * 80)
    print("MAXIMUM OCCUPATION NUMBER (n_max) CALCULATION")
    print("=" * 80)
    print(f"\nInput Parameters:")
    print(f"  Black hole mass: {bh_mass_solar} M☉")
    print(f"  Alpha: {alpha}")
    print(f"  Initial spin: {astar_init}")
    print(f"  Process: annihilation (l=m)")
    
    # Get all available annihilation levels
    levels = list(diff_power_ann_dict.keys())
    
    print(f"\nTotal levels to check: {len(levels)}")
    
    # Calculate n_max for each level
    results = []
    for level in levels:
        result = calculate_n_max_for_level(level, bh_mass_solar, alpha, astar_init)
        if result is not None:
            results.append(result)
    
    # Print results in table format
    print("\n" + "=" * 100)
    print(f"RESULTS: {len(results)} valid superradiant levels")
    print("=" * 100)
    
    print(f"\n{'Level':<8} {'n':<4} {'l':<4} {'m':<4} {'ω [eV]':<15} {'Δa*':<12} {'n_max':<15}")
    print("-" * 100)
    
    for res in results:
        level = res['level']
        n = res['n']
        l = res['l']
        m = res['m']
        omega = res['omega_eV']
        delta_astar = res['delta_astar']
        n_max = res['n_max']
        
        print(f"{level:<8} {n:<4} {l:<4} {m:<4} {omega:<15.6e} {delta_astar:<12.6f} {n_max:<15.6e}")
    
    print("=" * 100)
    
    # Summary statistics
    if results:
        n_max_values = [r['n_max'] for r in results]
        print(f"\nSummary Statistics:")
        print(f"  Maximum n_max: {max(n_max_values):.6e}")
        print(f"  Minimum n_max: {min(n_max_values):.6e}")
        print(f"  Mean n_max: {np.mean(n_max_values):.6e}")
        print(f"  Median n_max: {np.median(n_max_values):.6e}")


if __name__ == "__main__":
    main()
