"""
Detector Distance Reach Plotter

This module creates plots showing the distance reach of gravitational wave detectors
as a function of black hole mass for a given detection threshold and specific
gravitational atom processes (annihilation or transition).

Inputs:
    - Detection threshold (h_det): dimensionless strain
    - Process type: from ConvertedFunctions.py (e.g., '2p', '3d', '5f 4f')
    - Alpha (α): fine structure constant
    
Output:
    - Plot with:
        - X-axis: Black hole mass [M_☉]
        - Y-axis: Distance reach [kpc]
"""

import sys
import os

# Add path to access scripts from Sem 1
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '0. Scripts from Sem 1'))

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from ConvertedFunctions import diff_power_ann_dict, diff_power_trans_dict
from ParamCalculator import (
    calc_rg_from_bh_mass, calc_bh_mass, calc_mu_a,
    calc_omega_ann, calc_omega_transition,
    calc_total_power_ann, calc_total_power_trans,
    calc_annihilation_rate, calc_transition_rate,
    calc_n_max, calc_superradiance_rate,
    calc_detectable_radius_ann, calc_detectable_radius_trans,
    G_N
)


# ==================== Unit Conversion Functions ====================

def eV_inv_to_meters(r_eV_inv):
    """
    Convert distance from eV^-1 to meters.
    
    Args:
        r_eV_inv (float or np.ndarray): Distance [eV^-1]
    
    Returns:
        float or np.ndarray: Distance [m]
    """
    hbar = constants.hbar  # [J·s]
    c = constants.c  # [m/s]
    eV = constants.e  # [J/eV]
    return (hbar * c / eV) * r_eV_inv


def meters_to_kpc(r_m):
    """
    Convert distance from meters to kiloparsecs.
    
    Args:
        r_m (float or np.ndarray): Distance [m]
    
    Returns:
        float or np.ndarray: Distance [kpc]
    """
    parsec = 3.0857e16  # [m]
    kpc = 1000 * parsec  # [m]
    return r_m / kpc


def eV_inv_to_kpc(r_eV_inv):
    """
    Convert distance from eV^-1 to kiloparsecs.
    
    Args:
        r_eV_inv (float or np.ndarray): Distance [eV^-1]
    
    Returns:
        float or np.ndarray: Distance [kpc]
    """
    r_m = eV_inv_to_meters(r_eV_inv)
    return meters_to_kpc(r_m)


# ==================== Process Parsing Functions ====================

def parse_process(process):
    """
    Parse process string to determine if annihilation or transition.
    
    Args:
        process (str): Process specification (e.g., '2p', '3d', '5f 4f')
    
    Returns:
        tuple: (process_type, process_key, n_values)
            - process_type: 'annihilation' or 'transition'
            - process_key: key for the appropriate dictionary
            - n_values: principal quantum number(s) as int or tuple
    """
    process = process.strip()
    
    # Check if it's a transition (contains space)
    if ' ' in process:
        process_type = 'transition'
        process_key = process
        
        # Validate it exists
        if process_key not in diff_power_trans_dict:
            raise ValueError(f"Transition process '{process}' not found in ConvertedFunctions.py. "
                           f"Available transitions: {list(diff_power_trans_dict.keys())}")
        
        # Extract principal quantum numbers
        parts = process.split()
        n_e = int(parts[0][0])  # First digit of first part (excited state)
        n_g = int(parts[1][0])  # First digit of second part (ground state)
        n_values = (n_e, n_g)
        
    else:
        process_type = 'annihilation'
        process_key = process
        
        # Validate it exists
        if process_key not in diff_power_ann_dict:
            raise ValueError(f"Annihilation process '{process}' not found in ConvertedFunctions.py. "
                           f"Available annihilations: {list(diff_power_ann_dict.keys())}")
        
        # Extract principal quantum number
        n = int(process[0])  # First digit
        n_values = n
    
    return process_type, process_key, n_values


def extract_quantum_numbers(process):
    """
    Extract quantum numbers from process string.
    Enforces m=l (maximum azimuthal quantum number for the orbital).
    
    Args:
        process (str): Process specification (e.g., '2p', '5f', '7h 6h')
    
    Returns:
        dict: Dictionary with quantum numbers
            For annihilation: {'n': int, 'l': int, 'm': int} where m=l
            For transition: {'n_e': int, 'l_e': int, 'm_e': int, 'n_g': int, 'l_g': int, 'm_g': int} where m=l
    """
    orbital_to_l = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6, 'k': 7}
    
    if ' ' in process:
        # Transition
        parts = process.split()
        n_e = int(parts[0][0])
        l_e_char = parts[0][1]
        l_e = orbital_to_l[l_e_char]
        n_g = int(parts[1][0])
        l_g_char = parts[1][1]
        l_g = orbital_to_l[l_g_char]
        
        return {
            'n_e': n_e, 'l_e': l_e, 'm_e': l_e,  # m = l
            'n_g': n_g, 'l_g': l_g, 'm_g': l_g   # m = l
        }
    else:
        # Annihilation
        n = int(process[0])
        l_char = process[1]
        l = orbital_to_l[l_char]
        return {'n': n, 'l': l, 'm': l}  # m = l


# ==================== Distance Calculation Functions ====================

def calc_distance_reach_ann(h_det, alpha, process, bh_mass_solar, 
                            delta_a_star=0.01):
    """
    Calculate detector distance reach for annihilation process.
    Note: Automatically uses m=l (maximum azimuthal quantum number for the orbital).
    
    Args:
        h_det (float): Detection threshold strain (dimensionless)
        alpha (float): Fine structure constant (dimensionless)
        process (str): Annihilation process (e.g., '2p', '3d', '4f')
        bh_mass_solar (float): Black hole mass [M_☉]
        delta_a_star (float): Spin parameter difference (default: 0.01)
    
    Returns:
        float: Maximum detectable distance [kpc]
    """
    # Parse process
    process_type, process_key, n = parse_process(process)
    if process_type != 'annihilation':
        raise ValueError(f"Process '{process}' is not an annihilation process")
    
    # Extract quantum numbers (m=l enforced)
    qn = extract_quantum_numbers(process)
    m = qn['m']  # m = l
    
    # Calculate gravitational radius
    r_g = calc_rg_from_bh_mass(bh_mass_solar)
    
    # Calculate annihilation frequency
    omega_ann = calc_omega_ann(r_g, alpha, n)
    
    # Calculate annihilation rate
    ann_rate = calc_annihilation_rate(process_key, alpha, omega_ann, G_N=G_N, r_g=r_g)
    
    # Calculate n_max (using m=l)
    bh_mass_eV = r_g / G_N
    n_max = calc_n_max(bh_mass_eV, delta_a_star, m)
    
    # Calculate detectable radius in eV^-1
    r_max_eV_inv = calc_detectable_radius_ann(h_det, ann_rate, omega_ann, n_max, G_N=G_N)
    
    # Convert to kpc
    r_max_kpc = eV_inv_to_kpc(r_max_eV_inv)
    
    return r_max_kpc


def calc_distance_reach_trans(h_det, alpha, process, bh_mass_solar,
                               a_star=0.9):
    """
    Calculate detector distance reach for transition process.
    Note: Automatically uses m=l (maximum azimuthal quantum number for the orbital).
    
    Args:
        h_det (float): Detection threshold strain (dimensionless)
        alpha (float): Fine structure constant (dimensionless)
        process (str): Transition process (e.g., '3p 2p', '5f 4f')
        bh_mass_solar (float): Black hole mass [M_☉]
        a_star (float): Dimensionless spin parameter (default: 0.9)
    
    Returns:
        float: Maximum detectable distance [kpc]
    """
    # Parse process
    process_type, process_key, (n_e, n_g) = parse_process(process)
    if process_type != 'transition':
        raise ValueError(f"Process '{process}' is not a transition process")
    
    # Get quantum numbers (m=l enforced)
    qn = extract_quantum_numbers(process)
    l = qn['l_e']  # Use excited state l for superradiance
    m = qn['m_e']  # m = l for excited state
    n = n_e  # Use excited state n
    
    # Calculate gravitational radius
    r_g = calc_rg_from_bh_mass(bh_mass_solar)
    
    # Calculate transition frequency
    omega_trans = calc_omega_transition(r_g, alpha, n_e, n_g)
    
    # Calculate transition rate
    trans_rate = calc_transition_rate(process_key, alpha, omega_trans, G_N=G_N, r_g=r_g)
    
    # Calculate superradiance rate (using m=l)
    sr_rate = calc_superradiance_rate(l, m, n, a_star, r_g, alpha)
    
    # Calculate detectable radius in eV^-1
    r_max_eV_inv = calc_detectable_radius_trans(h_det, trans_rate, omega_trans, 
                                                 sr_rate, G_N=G_N)
    
    # Convert to kpc
    r_max_kpc = eV_inv_to_kpc(r_max_eV_inv)
    
    return r_max_kpc


# ==================== Plotting Functions ====================

def plot_distance_reach(h_det, alpha, process, bh_mass_range=None,
                        delta_a_star=0.01, a_star=0.9,
                        num_points=100, save_path=None, show_plot=True):
    """
    Create distance reach plot: detector reach [kpc] vs black hole mass [M_☉].
    Note: Automatically uses m=l (maximum azimuthal quantum number for the orbital).
    
    Args:
        h_det (float): Detection threshold strain (dimensionless)
        alpha (float): Fine structure constant (dimensionless)
        process (str): Process (e.g., '2p', '3d', '5f 4f')
        bh_mass_range (tuple): (min_mass, max_mass) in M_☉ (default: auto)
        delta_a_star (float): Spin parameter diff for annihilation (default: 0.01)
        a_star (float): Spin parameter for transition (default: 0.9)
        num_points (int): Number of points to plot (default: 100)
        save_path (str): Path to save figure (optional)
        show_plot (bool): Whether to display plot (default: True)
    
    Returns:
        tuple: (bh_masses, distances) - arrays of BH masses and distances
    """
    # Parse process to determine type
    process_type, _, _ = parse_process(process)
    
    # Set default mass range if not provided
    if bh_mass_range is None:
        # Default ranges based on typical values for different alphas
        if alpha < 0.1:
            bh_mass_range = (0.1, 100)
        elif alpha < 0.3:
            bh_mass_range = (0.01, 10)
        else:
            bh_mass_range = (0.001, 1)
    
    # Generate black hole mass array (log scale)
    bh_masses = np.logspace(np.log10(bh_mass_range[0]), 
                           np.log10(bh_mass_range[1]), 
                           num_points)
    
    # Calculate distances
    distances = np.zeros_like(bh_masses)
    
    for i, bh_mass in enumerate(bh_masses):
        try:
            if process_type == 'annihilation':
                distances[i] = calc_distance_reach_ann(
                    h_det, alpha, process, bh_mass, 
                    delta_a_star=delta_a_star
                )
            else:  # transition
                distances[i] = calc_distance_reach_trans(
                    h_det, alpha, process, bh_mass,
                    a_star=a_star
                )
        except Exception as e:
            print(f"Warning: Could not calculate distance for M_BH = {bh_mass:.3e} M_☉: {e}")
            distances[i] = np.nan
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.loglog(bh_masses, distances, linewidth=2, label=f'{process}')
    
    plt.xlabel(r'Black Hole Mass [$M_\odot$]', fontsize=14)
    plt.ylabel(r'Distance Reach [kpc]', fontsize=14)
    
    # Title with parameters
    title = f'Detector Distance Reach\n'
    title += f'Process: {process}, α = {alpha:.3f}, $h_{{det}}$ = {h_det:.2e}'
    if process_type == 'annihilation':
        title += f', Δa* = {delta_a_star:.3f}'
    else:
        title += f', a* = {a_star:.2f}'
    plt.title(title, fontsize=12)
    
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return bh_masses, distances


def plot_multiple_processes(h_det, alpha, processes, bh_mass_range=None,
                           delta_a_star=0.01, a_star=0.9,
                           num_points=100, save_path=None, show_plot=True):
    """
    Plot distance reach for multiple processes on the same plot.
    Note: Automatically uses m=l (maximum azimuthal quantum number for the orbital).
    
    Args:
        h_det (float): Detection threshold strain (dimensionless)
        alpha (float): Fine structure constant (dimensionless)
        processes (list): List of process strings
        bh_mass_range (tuple): (min_mass, max_mass) in M_☉ (default: auto)
        delta_a_star (float): Spin parameter diff for annihilation (default: 0.01)
        a_star (float): Spin parameter for transition (default: 0.9)
        num_points (int): Number of points to plot (default: 100)
        save_path (str): Path to save figure (optional)
        show_plot (bool): Whether to display plot (default: True)
    
    Returns:
        dict: Dictionary mapping process names to (bh_masses, distances) tuples
    """
    # Set default mass range if not provided
    if bh_mass_range is None:
        if alpha < 0.1:
            bh_mass_range = (0.1, 100)
        elif alpha < 0.3:
            bh_mass_range = (0.01, 10)
        else:
            bh_mass_range = (0.001, 1)
    
    # Generate black hole mass array (log scale)
    bh_masses = np.logspace(np.log10(bh_mass_range[0]), 
                           np.log10(bh_mass_range[1]), 
                           num_points)
    
    # Create plot
    plt.figure(figsize=(12, 7))
    
    results = {}
    
    for process in processes:
        distances = np.zeros_like(bh_masses)
        process_type, _, _ = parse_process(process)
        
        for i, bh_mass in enumerate(bh_masses):
            try:
                if process_type == 'annihilation':
                    distances[i] = calc_distance_reach_ann(
                        h_det, alpha, process, bh_mass,
                        delta_a_star=delta_a_star
                    )
                else:  # transition
                    distances[i] = calc_distance_reach_trans(
                        h_det, alpha, process, bh_mass,
                        a_star=a_star
                    )
            except Exception as e:
                distances[i] = np.nan
        
        plt.loglog(bh_masses, distances, linewidth=2, label=f'{process}')
        results[process] = (bh_masses, distances)
    
    plt.xlabel(r'Black Hole Mass [$M_\odot$]', fontsize=14)
    plt.ylabel(r'Distance Reach [kpc]', fontsize=14)
    
    title = f'Detector Distance Reach Comparison\n'
    title += f'α = {alpha:.3f}, $h_{{det}}$ = {h_det:.2e}'
    plt.title(title, fontsize=12)
    
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return results


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Detector Distance Reach Plotter - Test Suite")
    print("=" * 70)
    
    # Test parameters
    h_det = 1e-37  # Detection threshold
    alpha = 0.1    # Fine structure constant
    bh_mass_range = (1e-12, 1e-5)
    
    print(f"\nTest Parameters:")
    print(f"  Detection threshold (h_det): {h_det:.2e}")
    print(f"  Fine structure constant (α): {alpha:.3f}")
    print()
    
    # Test 1: Single annihilation process
    print("-" * 70)
    print("Test 1: Annihilation Process (2p)")
    print("-" * 70)
    try:
        bh_masses, distances = plot_distance_reach(
            h_det=h_det,
            alpha=alpha,
            process='2p',
            bh_mass_range=bh_mass_range,
            num_points=50,
            save_path='distance_reach_2p_annihilation.png',
            show_plot=False
        )
        print(f"✓ Successfully plotted 2p annihilation")
        print(f"  Mass range: {bh_masses[0]:.2e} to {bh_masses[-1]:.2e} M_☉")
        valid_distances = distances[~np.isnan(distances)]
        if len(valid_distances) > 0:
            print(f"  Distance range: {np.min(valid_distances):.2e} to {np.max(valid_distances):.2e} kpc")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()
    
    # Test 2: Single transition process
    print("-" * 70)
    print("Test 2: Transition Process (6f 5f)")
    print("-" * 70)
    try:
        bh_masses, distances = plot_distance_reach(
            h_det=h_det,
            alpha=alpha,
            a_star=0.687,
            process='6f 5f',
            bh_mass_range=bh_mass_range,
            num_points=50,
            save_path='distance_reach_6f5f_transition.png',
            show_plot=False
        )
        print(f"✓ Successfully plotted 3p 2p transition")
        print(f"  Mass range: {bh_masses[0]:.2e} to {bh_masses[-1]:.2e} M_☉")
        valid_distances = distances[~np.isnan(distances)]
        if len(valid_distances) > 0:
            print(f"  Distance range: {np.min(valid_distances):.2e} to {np.max(valid_distances):.2e} kpc")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()
    
    # Test 3: Multiple annihilation processes
    print("-" * 70)
    print("Test 3: Multiple Annihilation Processes")
    print("-" * 70)
    try:
        processes = ['2p', '3p', '3d']
        results = plot_multiple_processes(
            h_det=h_det,
            alpha=alpha,
            processes=processes,
            bh_mass_range=bh_mass_range,
            num_points=50,
            save_path='distance_reach_multiple_annihilation.png',
            show_plot=False
        )
        print(f"✓ Successfully plotted {len(processes)} annihilation processes")
        for proc in processes:
            masses, dists = results[proc]
            valid = dists[~np.isnan(dists)]
            if len(valid) > 0:
                print(f"  {proc}: {np.min(valid):.2e} to {np.max(valid):.2e} kpc")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()
    
    # Test 4: Calculate specific point
    print("-" * 70)
    print("Test 4: Single Point Calculation")
    print("-" * 70)
    try:
        bh_mass = 1.0e-6  # Solar mass
        process = '2p'
        distance = calc_distance_reach_ann(h_det, alpha, process, bh_mass)
        print(f"✓ For M_BH = {bh_mass} M_☉, process {process}:")
        print(f"  Distance reach: {distance:.4e} kpc")
        print(f"  Distance reach: {distance * 1000:.4e} pc")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()
    print("=" * 70)
    print("Test suite completed!")
    print("=" * 70)
