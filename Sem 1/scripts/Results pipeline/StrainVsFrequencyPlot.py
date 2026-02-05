"""
Plot peak gravitational wave strain vs frequency for annihilation or transition processes.

This script takes black hole mass, alpha, and process type as inputs and calculates the peak
strain for all available processes (annihilations or transitions).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from ParamCalculator import (
    calc_rg_from_bh_mass,
    calc_omega_ann,
    calc_omega_transition,
    calc_annihilation_rate,
    calc_transition_rate,
    calc_h_peak_ann,
    calc_h_peak_trans,
    calc_delta_astar,
    calc_n_max,
    calc_superradiance_rate,
    calc_char_t_ann,
    calc_char_t_tran,
    G_N
)
from ConvertedFunctions import diff_power_ann_dict, diff_power_trans_dict


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


def parse_transition(trans_str):
    """
    Parse transition string (e.g., '3p 2p', '6g 5g') into quantum numbers.
    
    Args:
        trans_str (str): Transition identifier (e.g., '3p 2p', '6g 5g')
    
    Returns:
        tuple: (n_e, m_e, n_g, m_g) where _e is excited state and _g is ground state
    """
    parts = trans_str.split()
    if len(parts) != 2:
        raise ValueError(f"Invalid transition format: {trans_str}")
    
    n_e, m_e = parse_level(parts[0])
    n_g, m_g = parse_level(parts[1])
    return n_e, m_e, n_g, m_g


def calculate_strain_for_level(level_str, bh_mass_solar, alpha, astar_init, distance, debug=False):
    """
    Calculate peak strain, annihilation rate, and superradiance rate for a specific annihilation level.
    
    Args:
        level_str (str): Level identifier (e.g., '2p', '3d')
        bh_mass_solar (float): Black hole mass [M_☉]
        alpha (float): Fine structure constant (dimensionless)
        astar_init (float): Initial spin parameter (dimensionless)
        distance (float): Distance to source [eV^-1]
        debug (bool, optional): Print debug info. Defaults to False.
    
    Returns:
        tuple: (frequency [GHz], peak strain, rate [eV], superradiance rate [eV]) or (None, None, None, None) if invalid
    """
    # Parse level
    n, m = parse_level(level_str)
    
    # Calculate gravitational radius
    r_g = calc_rg_from_bh_mass(bh_mass_solar)  # [eV^-1]
    
    # Calculate black hole mass in eV
    bh_mass_eV = r_g / G_N  # [eV]
    
    # Calculate frequency
    omega = calc_omega_ann(r_g, alpha, n)  # [eV]
    omega_GHz = omega / 4.135667696e-6  # Convert to GHz
    
    # Calculate spin parameter change
    delta_astar = calc_delta_astar(astar_init, r_g, alpha, n, m)
    
    if debug:
        print(f"    n={n}, m={m}, r_g={r_g:.3e}, omega={omega:.3e} eV, delta_astar={delta_astar:.6f}")
    
    # Check if level is valid (delta_astar should be positive)
    if delta_astar <= 0:
        if debug:
            print(f"    INVALID: delta_astar <= 0")
        return None, None, None, None, None
    
    # Calculate maximum occupation number
    n_max = calc_n_max(bh_mass_eV, delta_astar, m)
    
    if debug:
        print(f"    n_max={n_max:.3e}")
    
    # Calculate annihilation rate
    ann_rate = calc_annihilation_rate(level_str, alpha, omega, G_N, r_g)  # [eV]
    
    if debug:
        print(f"    ann_rate={ann_rate:.3e} eV")
    
    # Calculate peak strain
    h_peak = calc_h_peak_ann(ann_rate, omega, distance, n_max)
    
    # Calculate superradiance rate (for annihilation: l = m)
    sr_rate = calc_superradiance_rate(m, m, n, astar_init, r_g, alpha)  # [eV]
    
    if debug:
        print(f"    h_peak={h_peak:.3e}")
        print(f"    sr_rate={sr_rate:.3e} eV")
    
    return omega_GHz, h_peak, ann_rate, sr_rate, n_max


def calculate_strain_for_transition(trans_str, bh_mass_solar, alpha, astar_init, distance, debug=False):
    """
    Calculate peak strain, transition rate, and superradiance rate for a specific transition.
    
    Args:
        trans_str (str): Transition identifier (e.g., '3p 2p', '6g 5g')
        bh_mass_solar (float): Black hole mass [M_☉]
        alpha (float): Fine structure constant (dimensionless)
        astar_init (float): Initial spin parameter (dimensionless)
        distance (float): Distance to source [eV^-1]
        debug (bool, optional): Print debug info. Defaults to False.
    
    Returns:
        tuple: (frequency [GHz], peak strain, rate [eV], superradiance rate [eV]) or (None, None, None, None) if invalid
    """
    # Parse transition
    n_e, m_e, n_g, m_g = parse_transition(trans_str)
    
    # For transitions, use the excited state (higher level) for superradiance calculation
    n = n_e
    m = m_e
    
    # Calculate gravitational radius
    r_g = calc_rg_from_bh_mass(bh_mass_solar)  # [eV^-1]
    
    # Calculate frequency
    omega = calc_omega_transition(r_g, alpha, n_e, n_g)  # [eV]
    omega_GHz = omega / 4.135667696e-6  # Convert to GHz
    
    # Calculate spin parameter change
    delta_astar = calc_delta_astar(astar_init, r_g, alpha, n, m)
    
    if debug:
        print(f"    n_e={n_e}, m_e={m_e}, n_g={n_g}, m_g={m_g}, omega={omega:.3e} eV, delta_astar={delta_astar:.6f}")
    
    # Check if level is valid (delta_astar should be positive)
    if delta_astar <= 0:
        if debug:
            print(f"    INVALID: delta_astar <= 0")
        return None, None, None, None, None
    
    # Calculate superradiance rate (for excited state: l = m)
    sr_rate = calc_superradiance_rate(m, m, n, astar_init, r_g, alpha)  # [eV]
    
    if debug:
        print(f"    sr_rate={sr_rate:.3e} eV")
    
    # Calculate transition rate
    trans_rate = calc_transition_rate(trans_str, alpha, omega, G_N, r_g)  # [eV]
    
    if debug:
        print(f"    trans_rate={trans_rate:.3e} eV")
    
    # Calculate peak strain
    h_peak = calc_h_peak_trans(trans_rate, omega, distance, sr_rate)
    
    if debug:
        print(f"    h_peak={h_peak:.3e}")
    
    return omega_GHz, h_peak, trans_rate, sr_rate, None  # None for n_max (not used in transitions)


def plot_strain_vs_frequency(bh_mass_solar, alpha, plot_type='strain', process='annihilation', astar_init=0.687, distance_kpc=10, exclude_processes=None):
    """
    Create scatter plot of peak strain or rate vs frequency for all processes.
    
    Args:
        bh_mass_solar (float): Black hole mass [M_☉]
        alpha (float): Fine structure constant (dimensionless)
        plot_type (str, optional): 'strain' or 'rate' to plot peak strain or process rate. Defaults to 'strain'.
        process (str, optional): 'annihilation' or 'transition'. Defaults to 'annihilation'.
        astar_init (float, optional): Initial spin parameter. Defaults to 0.687.
        distance_kpc (float, optional): Distance to source [kpc]. Defaults to 10.
        exclude_processes (list, optional): List of process labels to exclude from plot. 
            For annihilations: e.g., ["5g", "2p"]
            For transitions: e.g., ["3p 2p", "7f 6f", "6g 5g"]. Defaults to None.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

    # Convert distance to natural units
    distance_m = distance_kpc * 3.086e19  # [m]
    distance = distance_m / (1.9732705e-7)  # [eV^-1]
    
    # Initialize exclude list if not provided
    if exclude_processes is None:
        exclude_processes = []
    
    # Get all available processes based on type
    if process == 'annihilation':
        processes = list(diff_power_ann_dict.keys())
        calc_func = calculate_strain_for_level
        rate_name = 'Annihilation'
    else:  # transition
        processes = list(diff_power_trans_dict.keys())
        calc_func = calculate_strain_for_transition
        rate_name = 'Transition'
    
    # Filter out excluded processes
    processes = [p for p in processes if p not in exclude_processes]
    
    # Calculate strain and rate for each process
    frequencies = []
    strains = []
    rates = []
    sr_rates = []
    char_times = []
    valid_processes = []
    
    print(f"\nChecking {len(processes)} {process}s...")
    for proc in processes:
        freq, strain, rate, sr_rate, n_max = calc_func(
            proc, bh_mass_solar, alpha, astar_init, distance, debug=False
        )
        if freq is not None and strain is not None and strain > 0:
            frequencies.append(freq)
            strains.append(strain)
            # Convert rates from eV to yr^-1
            rates.append(rate * 31556926 / 6.582119569e-16)
            sr_rates.append(sr_rate * 31556926 / 6.582119569e-16)
            
            # Calculate characteristic time in years
            if process == 'annihilation':
                char_t = calc_char_t_ann(rate, n_max)  # [eV^-1]
            else:  # transition
                char_t = calc_char_t_tran(sr_rate)  # [eV^-1]
            # Convert from eV^-1 to years: t[yr] = t[eV^-1] / (31556926 / 6.582119569e-16)
            char_t_yr = char_t / (31556926 / 6.582119569e-16)
            char_times.append(char_t_yr)
            print(f"Process: {proc}, Char_time (years): {char_t_yr}, Char_time (seconds): {char_t_yr * 31556926}")
            
            valid_processes.append(proc)
    
    if len(valid_processes) == 0:
        print("\nERROR: No valid processes found!")
        print("This may be due to:")
        print("  - Alpha too large (must be < 1)")
        print("  - Black hole mass too small/large")
        print("  - Spin parameter out of range")
        return None, [], [], [], []
    
    # Choose y-axis data based on plot_type
    if plot_type == 'rate':
        y_data = rates
        y_label = f'{rate_name} Rate [yr$^{{-1}}$]'
        title_text = f'{rate_name} Rate vs Frequency for {rate_name} Processes'
        color_data = sr_rates
        color_label = 'Superradiance Rate $\\Gamma$ [yr$^{-1}$]'
    else:
        y_data = strains
        y_label = 'Peak Strain $h_\\mathrm{peak}$'
        title_text = f'Peak GW Strain vs Frequency for {rate_name} Processes'
        color_data = char_times
        color_label = 'Characteristic Time $\\tau$ [yr]'
    
    # Create plot with color coding
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(frequencies, y_data, c=color_data, s=50, alpha=0.8, 
                         cmap='viridis',
                         norm=plt.matplotlib.colors.LogNorm())
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(color_label, fontsize=17)
    cbar.ax.tick_params(labelsize=12)
    
    # Label each point
    for freq, y_val, proc in zip(frequencies, y_data, valid_processes):
        # Format label based on process type
        if process == 'transition':
            # Convert "3p 2p" to "$3p \rightarrow 2p$"
            parts = proc.split()
            if len(parts) == 2:
                label = f"${parts[0]} \\rightarrow {parts[1]}$"
            else:
                label = proc
        else:
            # Annihilation labels remain as-is
            label = proc
        
        # Shift p labels up, others down
        if process == 'transition':
            # Check if transition starts at n=7 level
            if proc.startswith('7') and not proc == "7f 4f":
                xytext = (-8, 5)  # Shift left for n=7 transitions
                ha = 'right'
            else:
                xytext = (0, 5)  # Directly above for other transitions
                ha = 'center'
        else:
            xytext = (3, 5)  # Offset for annihilations
            ha = 'left'
        plt.annotate(label, (freq, y_val), 
                    textcoords="offset points", xytext=xytext, 
                    ha=ha, va='center', fontsize=11.5, color='black')
    
    plt.xlabel('Frequency [GHz]', fontsize=17)
    plt.ylabel(y_label, fontsize=17)
    plt.yscale('log')
    
    # Use log scale for x-axis if plotting transitions
    if process == 'transition':
        plt.xscale('log')
    
    # Reduce ticks and increase tick label size
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Configure x-axis formatting based on scale
    if process == 'transition':
        # For log scale, use log locator
        ax.xaxis.set_major_locator(plt.matplotlib.ticker.LogLocator(base=10, numticks=8))
    else:
        # For linear scale, use existing formatting
        ax.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(6))
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    ax.yaxis.set_major_locator(plt.matplotlib.ticker.LogLocator(base=10, numticks=6))
    cbar.ax.yaxis.set_major_locator(plt.matplotlib.ticker.LogLocator(base=10, numticks=6))
    
    # Add small margins to prevent label clipping
    ax.margins(x=0.1, y=0.05)
    
    plt.grid(True, alpha=0.3, which='both', linestyle='--')
    plt.tight_layout()
    
    # Save the plot as PDF with descriptive filename
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd(), 'Final plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    if plot_type == 'rate':
        plot_name = f'{rate_name}RateVsFrequency'
    else:
        plot_name = f'{rate_name}StrainVsFrequency'
    
    filename = f"{plot_name}_M{bh_mass_solar:.2e}_alpha{alpha:.2f}_astar{astar_init:.3f}_d{distance_kpc}kpc.pdf"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    print(f"\nPlot saved to: {filepath}")
    
    return plt.gcf(), frequencies, strains, rates, valid_processes


if __name__ == "__main__":
    # Input parameters
    bh_mass_solar = 1e-6  # Black hole mass in solar masses
    alpha = 0.1         # Fine structure constant (typical range: 0.01 - 0.5)
    plot_type = 'strain' # 'strain' or 'rate' - what to plot on y-axis
    process = 'transition'  # 'annihilation' or 'transition'
    astar_init = 0.687    # Initial spin parameter
    distance_kpc = 10     # Distance in kpc
    
    # Exclude specific processes (optional)
    # For annihilations: e.g., ["5g", "2p"]
    # For transitions: e.g., ["3p 2p", "7f 6f", "6g 5g"]
    exclude_processes = ["5p 2p", "6p 2p", "6p 3p", "6d 3d", "7p 5p", "7d 5d", "7f 5f", "6f 4f", "6d 4d", "6p 4p", "7p 4p", "7d 4d", "4p 2p", "4p 3p", "7p 3p", "5d 3d"]  # Example for transitions
    
    # Create plot
    print(f"Calculating peak strain and rates for all {process}s...")
    print(f"Black hole mass: {bh_mass_solar} M_sun")
    print(f"Alpha: {alpha}")
    print(f"Process type: {process}")
    print(f"Plot type: {plot_type}")
    print(f"Initial spin: {astar_init}")
    print(f"Distance: {distance_kpc} kpc")
    if exclude_processes:
        print(f"Excluding processes: {exclude_processes}")
    print()
    print("NOTE: Alpha must be small enough that astar_init > astar_crit for superradiance.")
    print()
    
    fig, freqs, strains, rates, processes_list = plot_strain_vs_frequency(
        bh_mass_solar, alpha, plot_type, process, astar_init, distance_kpc, exclude_processes
    )
    
    if fig is None:
        print("\nPlot generation failed. Exiting.")
    else:
        # Print results
        print(f"\nValid {process}s found: {len(processes_list)}")
        print("\nResults:")
        
        if process == 'annihilation':
            rate_label = 'Ann. Rate [yr^-1]'
        else:
            rate_label = 'Trans. Rate [yr^-1]'
        
        print(f"{'Process':<12} {'Frequency [GHz]':<18} {'Peak Strain':<15} {rate_label:<20}")
        print("-" * 70)
        for proc, freq, strain, rate in zip(processes_list, freqs, strains, rates):
            print(f"{proc:<12} {freq:<18.6e} {strain:<15.6e} {rate:<20.6e}")
        
        plt.show()
