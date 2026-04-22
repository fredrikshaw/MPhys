"""
Plot peak gravitational wave strain vs frequency for annihilation or transition processes.

This script takes black hole mass, alpha, and process type as inputs and calculates the peak
strain for all available processes (annihilations or transitions).

plot_type options:
    'strain'  — plots h_peak
    'rate'    — plots the process rate
    'signal'  — plots h_peak * sqrt(tau), the signal prospect figure of merit
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sem2_dir = None
for p in current_dir.parents:
    if p.name == "Sem 2":
        sem2_dir = p
        break
if sem2_dir is None:
    sem2_dir = current_dir.parent

script_dir = sem2_dir / "0. Scripts from Sem 1"
if not script_dir.exists():
    for p in current_dir.parents:
        candidate = p / "0. Scripts from Sem 1"
        if candidate.exists():
            script_dir = candidate
            break
sys.path.append(str(script_dir.resolve()))

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

    Returns:
        tuple: (frequency [GHz], peak strain, rate [eV], superradiance rate [eV], n_max)
               or (None, None, None, None, None) if invalid
    """
    n, m = parse_level(level_str)

    r_g        = calc_rg_from_bh_mass(bh_mass_solar)   # [eV^-1]
    bh_mass_eV = r_g / G_N                              # [eV]

    omega      = calc_omega_ann(r_g, alpha, n)          # [eV]
    omega_GHz  = omega / 4.135667696e-6

    delta_astar = calc_delta_astar(astar_init, r_g, alpha, n, m)

    if debug:
        print(f"    n={n}, m={m}, r_g={r_g:.3e}, omega={omega:.3e} eV, delta_astar={delta_astar:.6f}")

    if delta_astar <= 0:
        if debug:
            print(f"    INVALID: delta_astar <= 0")
        return None, None, None, None, None

    n_max    = calc_n_max(bh_mass_eV, delta_astar, m)
    ann_rate = calc_annihilation_rate(level_str, alpha, omega, G_N, r_g)   # [eV]
    h_peak   = calc_h_peak_ann(ann_rate, omega, distance, n_max)
    sr_rate  = calc_superradiance_rate(m, m, n, astar_init, r_g, alpha)    # [eV]

    if debug:
        print(f"    n_max={n_max:.3e}")
        print(f"    ann_rate={ann_rate:.3e} eV")
        print(f"    h_peak={h_peak:.3e}")
        print(f"    sr_rate={sr_rate:.3e} eV")

    return omega_GHz, h_peak, ann_rate, sr_rate, n_max


def calculate_strain_for_transition(trans_str, bh_mass_solar, alpha, astar_init, distance, debug=False):
    """
    Calculate peak strain, transition rate, and superradiance rate for a specific transition.

    Returns:
        tuple: (frequency [GHz], peak strain, rate [eV], superradiance rate [eV], None)
               or (None, None, None, None, None) if invalid
    """
    n_e, m_e, n_g, m_g = parse_transition(trans_str)
    n = n_e
    m = m_e

    r_g     = calc_rg_from_bh_mass(bh_mass_solar)          # [eV^-1]
    omega   = calc_omega_transition(r_g, alpha, n_e, n_g)   # [eV]
    omega_GHz = omega / 4.135667696e-6

    delta_astar = calc_delta_astar(astar_init, r_g, alpha, n, m)

    if debug:
        print(f"    n_e={n_e}, m_e={m_e}, n_g={n_g}, m_g={m_g}, omega={omega:.3e} eV, delta_astar={delta_astar:.6f}")

    if delta_astar <= 0:
        if debug:
            print(f"    INVALID: delta_astar <= 0")
        return None, None, None, None, None

    sr_rate    = calc_superradiance_rate(m, m, n, astar_init, r_g, alpha)  # [eV]
    trans_rate = calc_transition_rate(trans_str, alpha, omega, G_N, r_g)   # [eV]
    h_peak     = calc_h_peak_trans(trans_rate, omega, distance, sr_rate)

    if debug:
        print(f"    sr_rate={sr_rate:.3e} eV")
        print(f"    trans_rate={trans_rate:.3e} eV")
        print(f"    h_peak={h_peak:.3e}")

    return omega_GHz, h_peak, trans_rate, sr_rate, None


def plot_strain_vs_frequency(bh_mass_solar, alpha, plot_type='strain', process='annihilation',
                              astar_init=0.687, distance_kpc=10, exclude_processes=None):
    """
    Create scatter plot of peak strain, rate, or signal prospect (h*sqrt(tau)) vs frequency.

    Args:
        bh_mass_solar   : float  — Black hole mass [M_☉]
        alpha           : float  — Fine structure constant (dimensionless)
        plot_type       : str    — 'strain', 'rate', or 'signal'
        process         : str    — 'annihilation' or 'transition'
        astar_init      : float  — Initial spin parameter
        distance_kpc    : float  — Distance to source [kpc]
        exclude_processes : list — Process labels to exclude
    """
    plt.rcParams.update({
        "text.usetex"         : True,
        "font.family"         : "serif",
        "font.serif"          : ["Computer Modern Roman"],
        "text.latex.preamble" : r"\usepackage{amsmath}",
        "font.size"           : 16,
        "axes.titlesize"      : 16,
        "axes.labelsize"      : 15,
        "xtick.labelsize"     : 14,
        "ytick.labelsize"     : 14,
        "legend.fontsize"     : 14,
        "figure.titlesize"    : 18,
    })

    # Convert distance to natural units [eV^-1]
    distance_m = distance_kpc * 3.086e19
    distance   = distance_m / 1.9732705e-7

    if exclude_processes is None:
        exclude_processes = []

    if process == 'annihilation':
        processes  = list(diff_power_ann_dict.keys())
        calc_func  = calculate_strain_for_level
        rate_name  = 'Annihilation'
    else:
        processes  = list(diff_power_trans_dict.keys())
        calc_func  = calculate_strain_for_transition
        rate_name  = 'Transition'

    processes = [p for p in processes if p not in exclude_processes]

    # ── Collect data ──────────────────────────────────────────────────────────
    frequencies    = []
    strains        = []
    rates          = []
    sr_rates       = []
    char_times     = []   # [yr]
    signal_prospects = []  # h_peak * sqrt(tau)  [yr^(1/2)]
    valid_processes  = []

    EV_PER_YEAR = 31556926 / 6.582119569e-16   # [eV / yr^-1]  i.e. 1 eV^-1 in years

    print(f"\nChecking {len(processes)} {process}s...")

    for proc in processes:
        freq, strain, rate, sr_rate, n_max = calc_func(
            proc, bh_mass_solar, alpha, astar_init, distance, debug=False
        )
        if freq is None or strain is None or strain <= 0:
            continue

        frequencies.append(freq)
        strains.append(strain)

        # Rates in yr^-1
        rates.append(rate * EV_PER_YEAR)
        sr_rates.append(sr_rate * EV_PER_YEAR)

        # Characteristic time in years [eV^-1 -> yr]
        if process == 'annihilation':
            char_t_eV = calc_char_t_ann(rate, n_max)   # [eV^-1]
        else:
            char_t_eV = calc_char_t_tran(sr_rate)      # [eV^-1]

        char_t_yr = char_t_eV / EV_PER_YEAR            # [yr]
        char_times.append(char_t_yr)

        # ── Signal prospect: h_peak * sqrt(tau [yr]) ─────────────────────────
        # This is the numerator of the SNR expression and directly proportional
        # to the detector distance reach. Larger h*sqrt(tau) -> better detection.
        signal_prospects.append(strain * np.sqrt(abs(char_t_yr)))

        print(f"  {proc:12s}  f={freq:.4e} GHz  h={strain:.3e}  "
              f"tau={char_t_yr:.3e} yr  h*sqrt(tau)={signal_prospects[-1]:.3e} yr^(1/2)")

        valid_processes.append(proc)

    if len(valid_processes) == 0:
        print("\nERROR: No valid processes found!")
        return None, [], [], [], [], []

    # ── Choose y-axis data ────────────────────────────────────────────────────
    if plot_type == 'rate':
        y_data      = rates
        y_label     = fr'{rate_name} Rate [yr$^{{-1}}$]'
        color_data  = sr_rates
        color_label = r'Superradiance Rate $\Gamma$ [yr$^{-1}$]'
        plot_name   = f'{rate_name}RateVsFrequency'

    elif plot_type == 'signal':
        y_data      = signal_prospects
        y_label     = r'$h_0\,\sqrt{\tau}$ [yr$^{1/2}$]'
        color_data  = char_times
        color_label = r'Characteristic Time $\tau$ [yr]'
        plot_name   = f'{rate_name}SignalProspectVsFrequency'

    else:   # 'strain'
        y_data      = strains
        y_label     = r'Peak Strain $h_\mathrm{peak}$'
        color_data  = char_times
        color_label = r'Characteristic Time $\tau$ [yr]'
        plot_name   = f'{rate_name}StrainVsFrequency'

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.figure(figsize=(5, 5))
    scatter = plt.scatter(frequencies, y_data,
                          c=color_data, s=50, alpha=0.8,
                          cmap='viridis',
                          norm=plt.matplotlib.colors.LogNorm())

    cbar = plt.colorbar(scatter)
    cbar.set_label(color_label, fontsize=17)
    cbar.ax.tick_params(labelsize=12)

    # ── Point labels ──────────────────────────────────────────────────────────
    for freq, y_val, proc in zip(frequencies, y_data, valid_processes):
        if process == 'transition':
            parts = proc.split()
            label = fr"${parts[0]} \rightarrow {parts[1]}$" if len(parts) == 2 else proc
        else:
            label = proc

        if process == 'transition':
            if proc.startswith('7') and proc != "7f 4f":
                xytext = (-8, 5);  ha = 'right'
            else:
                xytext = (0, 5);   ha = 'center'
        elif 'd' in proc:
            xytext = (1, -6);  ha = 'left'
        else:
            xytext = (1, 6);   ha = 'left'

        plt.annotate(label, (freq, y_val),
                     textcoords="offset points", xytext=xytext,
                     ha=ha, va='center', fontsize=12.5, color='black')

    # ── Axes formatting ───────────────────────────────────────────────────────
    plt.xlabel('Frequency [GHz]', fontsize=17)
    plt.ylabel(y_label, fontsize=17)
    plt.yscale('log')

    if process == 'transition':
        plt.xscale('log')

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=12)
    for lbl in ax.get_xticklabels()[::2]:
        lbl.set_visible(False)

    if process == 'transition':
        ax.xaxis.set_major_locator(plt.matplotlib.ticker.LogLocator(base=10, numticks=8))
    else:
        ax.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(6))
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

    ax.yaxis.set_major_locator(plt.matplotlib.ticker.LogLocator(base=10, numticks=6))
    cbar.ax.yaxis.set_major_locator(plt.matplotlib.ticker.LogLocator(base=10, numticks=6))
    ax.margins(x=0.1, y=0.05)

    plt.grid(True, alpha=0.3, which='both', linestyle='--')
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    plot_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__))
        if '__file__' in globals() else os.getcwd(),
        'Final plots'
    )
    os.makedirs(plot_dir, exist_ok=True)
    filename = (f"{plot_name}_M{bh_mass_solar:.2e}_alpha{alpha:.2f}"
                f"_astar{astar_init:.3f}_d{distance_kpc}kpc.pdf")
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    print(f"\nPlot saved to: {filepath}")

    return plt.gcf(), frequencies, strains, rates, valid_processes, char_times


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    bh_mass_solar     = 1e-11
    alpha             = 0.1
    plot_type         = 'signal'    # 'strain', 'rate', or 'signal'
    process           = 'transitions'
    astar_init        = 0.687
    distance_kpc      = 10
    exclude_processes = [
        "7h 6h", "6p 5p", "7g 6g", "6d 5d", "5f 4f", "7p 2p", "7d 3d",
        "8p 2p", "8p 3p", "8d 3d", "5p 2p", "6p 2p", "6p 3p", "6d 3d",
        "7p 5p", "7d 5d", "7f 5f", "6f 4f", "6d 4d", "6p 4p", "7p 4p",
        "7d 4d", "4p 2p", "4p 3p", "7p 3p", "5d 3d"
    ]

    print(f"Calculating signal prospects for all {process}s...")
    print(f"Black hole mass : {bh_mass_solar} M_sun")
    print(f"Alpha           : {alpha}")
    print(f"Process type    : {process}")
    print(f"Plot type       : {plot_type}")
    print(f"Initial spin    : {astar_init}")
    print(f"Distance        : {distance_kpc} kpc")
    if exclude_processes:
        print(f"Excluding       : {exclude_processes}")
    print()

    fig, freqs, strains, rates, processes_list, char_times = plot_strain_vs_frequency(
        bh_mass_solar, alpha, plot_type, process,
        astar_init, distance_kpc, exclude_processes
    )

    if fig is None:
        print("\nPlot generation failed.")
    else:
        print(f"\nValid {process}s found: {len(processes_list)}")
        print("\nResults:")

        rate_label = 'Ann. Rate [yr^-1]' if process == 'annihilation' else 'Trans. Rate [yr^-1]'
        print(f"{'Process':<12} {'Freq [GHz]':<16} {'h_peak':<14} "
              f"{'tau [yr]':<16} {'h*sqrt(tau)':<16}")
        print("-" * 80)

        SECONDS_IN_YEAR = 3.154e7
        for proc, freq, strain, char_t_yr in zip(
                processes_list, freqs, strains, char_times):
            signal = strain * np.sqrt(abs(char_t_yr))
            print(f"{proc:<12} {freq:<16.4e} {strain:<14.4e} "
                  f"{char_t_yr:<16.4e} {signal:<16.4e}")

        plt.show()