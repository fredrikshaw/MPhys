"""
Test script to calculate superradiance rates for excited states from n=2 to n=10.
Uses l=m constraint with fixed alpha and ParamCalculator functions.
"""
import sys
import os
import numpy as np

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, script_dir)

from ParamCalculator import (
    calc_rg_from_bh_mass,
    calc_omega_ann,
    calc_delta_astar,
    calc_superradiance_rate
)

def generate_level_list(max_n=10):
    """
    Generate list of level strings from 2p up to n=max_n.
    Uses standard spectroscopic notation (s, p, d, f, g, h, i, k, l, m, n, o, q, r, t).
    
    Args:
        max_n (int): Maximum principal quantum number
    
    Returns:
        list: List of level strings (e.g., ['2p', '3d', '3p', ...])
    """
    m_notation = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'q', 'r', 't']
    
    levels = []
    for n in range(2, max_n + 1):
        # For each n, m can range from 0 to n-1
        for m in range(0, n):
            if m < len(m_notation):
                level_str = f"{n}{m_notation[m]}"
                levels.append(level_str)
    
    return levels

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
    n = int(level_str[0]) if len(level_str) == 2 else int(level_str[:2])
    m = m_lookup[level_str[-1]]
    return n, m

def calculate_sr_rate_for_level(level_str, bh_mass_solar, alpha, astar_init):
    """
    Calculate superradiance rate for a given level using l=m constraint.
    
    Args:
        level_str (str): Level identifier (e.g., '2p', '3d')
        bh_mass_solar (float): Black hole mass [M_☉]
        alpha (float): Fine structure constant (dimensionless)
        astar_init (float): Initial spin parameter (dimensionless)
    
    Returns:
        dict: Dictionary with quantum numbers, alpha, frequency, and rates
    """
    # Parse level
    n, m = parse_level(level_str)
    
    # For l=m constraint
    l = m
    
    # Skip s orbitals (m=0, l=0) as they don't participate in superradiance
    if m == 0:
        return None
    
    # Calculate gravitational radius
    r_g = calc_rg_from_bh_mass(bh_mass_solar)  # [eV^-1]
    
    # Calculate frequency
    omega = calc_omega_ann(r_g, alpha, n)  # [eV]
    
    # Calculate spin parameter change
    delta_astar = calc_delta_astar(astar_init, r_g, alpha, n, m)
    astar_final = astar_init - delta_astar
    
    # Check if superradiance is possible
    if astar_final < 0 or delta_astar < 0:
        # Not superradiant
        return None
    
    # Calculate superradiance rate
    sr_rate = calc_superradiance_rate(l, m, n, astar_init, r_g, alpha)  # [eV]
    
    # Convert rate to inverse years using conversion from CalcFinalValues.py
    # Conversion: rate [yr^-1] = rate [eV] * 31556926 / 6.582119569e-16
    rate_inv_yr = sr_rate * 31556926 / 6.582119569e-16
    
    # Convert frequency to GHz for readability
    eV_to_Hz = 1.519e15  # 1 eV = 1.519e15 Hz
    freq_GHz = omega * eV_to_Hz / 1e9
    
    # Convert rate to inverse seconds for readability
    rate_inv_s = sr_rate * eV_to_Hz
    
    # Calculate timescale
    if sr_rate > 0:
        timescale_s = 1.0 / rate_inv_s
        timescale_yr = timescale_s / (365.25 * 24 * 3600)
    else:
        timescale_s = np.inf
        timescale_yr = np.inf
    
    return {
        'level': level_str,
        'n': n,
        'l': l,
        'm': m,
        'alpha': alpha,
        'omega_eV': omega,
        'freq_GHz': freq_GHz,
        'sr_rate_eV': sr_rate,
        'sr_rate_inv_s': rate_inv_s,
        'sr_rate_inv_yr': rate_inv_yr,
        'timescale_s': timescale_s,
        'timescale_yr': timescale_yr,
        'astar_init': astar_init,
        'astar_final': astar_final,
        'delta_astar': delta_astar
    }

def print_results_table(results, max_n):
    """
    Print results in a triangular table format with spectroscopic notation on vertical axis
    and n on horizontal axis. Shows two tables: rates in eV and rates in yr^-1.
    
    Args:
        results (list): List of result dictionaries
        max_n (int): Maximum principal quantum number
    """
    print("\n" + "="*140)
    print("SUPERRADIANCE RATES FOR EXCITED STATES (l=m constraint)")
    print("="*140)
    
    # Create a dictionary for quick lookup: (n, m) -> result
    result_dict = {(r['n'], r['m']): r for r in results}
    
    # Spectroscopic notation
    m_notation = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'q', 'r', 't']
    
    # Determine max m value from results
    max_m = max(r['m'] for r in results) if results else 0
    
    # Table 1: Superradiance Rate in eV
    print("\nSuperradiance Growth Rate Γ_SR [eV]")
    print("-"*140)
    
    # Print header with n values
    header = "     "
    for n in range(2, max_n + 1):
        header += f"n={n:<12} "
    print(header)
    print("-"*140)
    
    # Print each row (for each m/orbital type)
    for m in range(1, min(max_m + 1, len(m_notation))):  # Start from 1 to skip s orbitals
        orbital = m_notation[m]
        row = f"{orbital:<4} "
        
        for n in range(2, max_n + 1):
            if n > m:  # Only valid if n > l (and l=m)
                if (n, m) in result_dict:
                    rate_eV = result_dict[(n, m)]['sr_rate_eV']
                    row += f"{rate_eV:<14.4e} "
                else:
                    row += f"{'---':<14} "
            else:
                row += f"{'':<14} "
        
        print(row)
    
    print("-"*140)
    
    # Table 2: Superradiance Rate in yr^-1
    print("\n\nSuperradiance Growth Rate Γ_SR [yr⁻¹]")
    print("-"*140)
    
    # Print header with n values
    header = "     "
    for n in range(2, max_n + 1):
        header += f"n={n:<12} "
    print(header)
    print("-"*140)
    
    # Print each row
    for m in range(1, min(max_m + 1, len(m_notation))):
        orbital = m_notation[m]
        row = f"{orbital:<4} "
        
        for n in range(2, max_n + 1):
            if n > m:
                if (n, m) in result_dict:
                    rate_yr = result_dict[(n, m)]['sr_rate_inv_yr']
                    row += f"{rate_yr:<14.4e} "
                else:
                    row += f"{'---':<14} "
            else:
                row += f"{'':<14} "
        
        print(row)
    
    print("="*140)
    print(f"\nTotal superradiant modes found: {len(results)}")
    print(f"(Excluded s orbitals and non-superradiant modes)")

def main():
    """Main function to run the test."""
    # Input parameters
    bh_mass_solar = 1e-11  # Black hole mass in solar masses
    alpha = 0.1            # Fine structure constant (dimensionless)
    astar_init = 0.687     # Initial spin parameter
    max_n = 10             # Maximum principal quantum number
    
    print("="*120)
    print("SUPERRADIANCE RATE CALCULATION TEST")
    print("="*120)
    print(f"\nInput Parameters:")
    print(f"  Black hole mass: {bh_mass_solar} M☉")
    print(f"  Alpha: {alpha} (fixed for all states)")
    print(f"  Initial spin: {astar_init}")
    print(f"  Principal quantum number range: n = 2 to {max_n}")
    print(f"  Constraint: l = m (superradiance condition)")
    
    # Generate all levels
    levels = generate_level_list(max_n)
    print(f"\nTotal levels to check: {len(levels)}")
    
    # Calculate rates for all levels
    results = []
    for level in levels:
        result = calculate_sr_rate_for_level(level, bh_mass_solar, alpha, astar_init)
        if result is not None:
            results.append(result)
    
    # Print results table
    print_results_table(results, max_n)
    
    # Summary statistics
    if results:
        print(f"\n{'Summary Statistics:':<30}")
        print("-"*60)
        sr_rates = [r['sr_rate_eV'] for r in results]
        freqs = [r['freq_GHz'] for r in results]
        timescales = [r['timescale_yr'] for r in results if r['timescale_yr'] != np.inf]
        
        print(f"{'Fastest rate:':<30} {max(sr_rates):.6e} eV  ({max(sr_rates)*1.519e15:.6e} s⁻¹)")
        print(f"{'Slowest rate:':<30} {min(sr_rates):.6e} eV  ({min(sr_rates)*1.519e15:.6e} s⁻¹)")
        print(f"{'Frequency range:':<30} {min(freqs):.6e} - {max(freqs):.6e} GHz")
        if timescales:
            print(f"{'Shortest timescale:':<30} {min(timescales):.6e} years")
            print(f"{'Longest timescale:':<30} {max(timescales):.6e} years")

if __name__ == "__main__":
    main()
