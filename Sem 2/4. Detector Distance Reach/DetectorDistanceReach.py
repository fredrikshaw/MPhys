"""
Combined LIGO + MWB Detector Distance Reach
============================================

This script computes and plots the maximum detectable distance for superradiance
transitions for both:
- LIGO high-frequency sensitivity (Schnabel & Korobko 2024)
- Magnetic Weber Bar detectors (ADMX-EFR and DMRadio-GUT)
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Import MWB modules
from MagneticWeberBar import (
    MagneticWeberBar,
    ADMX_EFR,
    DMRADIO_GUT,
    G_NEWTON,
    C_LIGHT,
    HBAR,
    K_B,
    noise_equivalent_strain_broadband,
)

# Import LIGO module
from LIGOSensitivity import (
    DETECTORS,
    asd_power_law,
    get_ligo_noise_psd,
)

# ─────────────────────────────────────────────────────────────────────────────
# Directory resolution and imports for superradiance calculations
# ─────────────────────────────────────────────────────────────────────────────

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
    calc_omega_transition,
    calc_transition_rate,
    calc_rg_from_bh_mass,
    calc_h_peak_ann,
    calc_annihilation_rate,
    calc_omega_ann,
    calc_n_max,
    calc_bh_mass,
    calc_delta_astar,
    calc_char_t_ann
)

relativistic_dir = sem2_dir / "2. Relativistic Superradiance Rate"
if not relativistic_dir.exists():
    for p in current_dir.parents:
        candidate = p / "2. Relativistic Superradiance Rate"
        if candidate.exists():
            relativistic_dir = candidate
            break
sys.path.append(str(relativistic_dir.resolve()))

from SuperradianceRateCF import sr_rate_dimensioned

# ─────────────────────────────────────────────────────────────────────────────
# Unit conversion constants
# ─────────────────────────────────────────────────────────────────────────────

EV_TO_J      = 1.602176634e-19
HBAR_C_M_EV  = (HBAR * C_LIGHT) / EV_TO_J
EV_TO_SI     = EV_TO_J / HBAR
INV_EV_TO_M  = HBAR_C_M_EV
KPC_TO_M     = 3.086e19


# ─────────────────────────────────────────────────────────────────────────────
# Transition source factory
# ─────────────────────────────────────────────────────────────────────────────

def make_transition_funcs(alpha, transition, filepath):
    """
    Return (freq_func, h_func, tau_func) for a superradiance transition.
    """

    def freq_func(M_solar):
        r_g_nat     = calc_rg_from_bh_mass(M_solar)
        omega_t_nat = calc_omega_transition(r_g_nat, alpha, 4, 3)
        return omega_t_nat * EV_TO_SI / (2 * np.pi)

    def h_func(f_t, M_solar):
        r_g_nat     = calc_rg_from_bh_mass(M_solar)
        r_g_SI      = r_g_nat * INV_EV_TO_M
        Gamma_t_nat = calc_transition_rate(
                          transition, alpha, f_t, G_NEWTON, r_g_SI
                      )
        Gamma_t_SI  = Gamma_t_nat * EV_TO_SI
        Gamma_sr_SI = sr_rate_dimensioned(
                          alpha, M_solar, filepath=filepath, method='cf'
                      )['gamma_SI']
        omega_t_SI  = 2 * np.pi * f_t
        return np.sqrt(
            4 * G_NEWTON / omega_t_SI * Gamma_sr_SI**2 / Gamma_t_SI
        )

    def tau_func(f_t, M_solar):
        return 1.0 / sr_rate_dimensioned(
                         alpha, M_solar, filepath=filepath, method='cf'
                     )['gamma_SI']

    return freq_func, h_func, tau_func


# ─────────────────────────────────────────────────────────────────────────────
# Core computation for LIGO detector
# ─────────────────────────────────────────────────────────────────────────────

def compute_point_ligo(M_solar, freq_func, h_func, tau_func,
                       ligo_detector='adv_ligo', f_band=(3.75e4, 1e11), rho_star=1.0):
    """
    Compute distance reach for LIGO detector.
    
    For LIGO, we use the strain ASD directly from the power-law model.
    """
    try:
        f_t = freq_func(M_solar)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if not (f_band[0] <= f_t <= f_band[1]):
        return f_t, np.nan, np.nan, np.nan, np.nan

    try:
        h_unit = h_func(f_t, M_solar)
    except Exception:
        return f_t, np.nan, np.nan, np.nan, np.nan

    try:
        tau_val = tau_func(f_t, M_solar)
    except Exception:
        return f_t, np.nan, np.nan, np.nan, np.nan

    # Get LIGO noise PSD
    ligo_noise = get_ligo_noise_psd(np.array([f_t]), ligo_detector)[0]
    
    if np.isnan(ligo_noise):
        return f_t, np.nan, h_unit, tau_val, np.nan

    d_max_m   = h_unit * np.sqrt(tau_val / ligo_noise)
    d_max_kpc = d_max_m / KPC_TO_M

    if not np.isfinite(d_max_kpc) or d_max_kpc <= 0:
        return f_t, np.nan, h_unit, tau_val, ligo_noise

    return f_t, d_max_kpc, h_unit, tau_val, ligo_noise


def run_sweep_ligo(freq_func, h_func, tau_func,
                   M_range=(1e-12, 1e4), n_coarse=300, n_dense=800,
                   ligo_detector='adv_ligo', f_band=(3.75e4, 1e11), rho_star=1.0):
    """
    Mass sweep for LIGO detector.
    """
    def _sweep(M_array):
        f_arr, d_arr, M_arr = [], [], []
        for M in M_array:
            f, d, _, _, _ = compute_point_ligo(
                M, freq_func, h_func, tau_func, ligo_detector, f_band, rho_star
            )
            f_arr.append(f);  d_arr.append(d);  M_arr.append(M)
        return np.array(f_arr), np.array(d_arr), np.array(M_arr)

    M_coarse = np.logspace(np.log10(M_range[0]), np.log10(M_range[1]), n_coarse)
    f_coarse, d_coarse, M_coarse_arr = _sweep(M_coarse)
    
    # Find resonance peak
    valid = np.isfinite(d_coarse)
    M_res, f_res, d_res = np.nan, np.nan, np.nan
    d_fine = None
    M_fine = None
    
    if valid.any():
        # Find peak in coarse sweep
        idx_peak = np.argmax(d_coarse[valid])
        M_guess = M_coarse_arr[valid][idx_peak]
        
        # Dense sweep around peak
        M_dense = np.logspace(np.log10(M_guess/10), np.log10(M_guess*10), n_dense)
        f_dense, d_dense, _ = _sweep(M_dense)
        
        valid_dense = np.isfinite(d_dense)
        if valid_dense.any():
            M_fine = M_dense[valid_dense]
            d_fine = d_dense[valid_dense]
            idx_res = np.argmax(d_fine)
            M_res = M_fine[idx_res]
            f_res = f_dense[valid_dense][idx_res]
            d_res = d_fine[idx_res]
    
    # Combine coarse and dense
    valid_coarse = np.isfinite(d_coarse)
    if M_fine is not None:
        M_combined = np.concatenate([M_coarse_arr[valid_coarse], M_fine])
        d_combined = np.concatenate([d_coarse[valid_coarse], d_fine])
        sort_idx = np.argsort(M_combined)
        M_combined = M_combined[sort_idx]
        d_combined = d_combined[sort_idx]
    else:
        M_combined = M_coarse_arr[valid_coarse]
        d_combined = d_coarse[valid_coarse]
    
    return {
        'name': 'LIGO HF',
        'f': f_coarse[valid_coarse],
        'd': d_coarse[valid_coarse],
        'M': M_coarse_arr[valid_coarse],
        'd_combined': d_combined,
        'M_combined': M_combined,
        'f_res': f_res,
        'd_res': d_res,
        'M_res': M_res
    }


# ─────────────────────────────────────────────────────────────────────────────
# MWB sweep (using the existing run_sweep from MWBDetectorDistance)
# ─────────────────────────────────────────────────────────────────────────────

# Import run_sweep from MWBDetectorDistance
from MWBDetectorDistance import run_sweep as run_sweep_mwb


# ─────────────────────────────────────────────────────────────────────────────
# Plotting function combining all detectors
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined_reach(results_mwb, results_ligo, alpha, process_label,
                        savepath=None):
    """
    Plot distance reach for both MWB and LIGO detectors.
    
    Parameters
    ----------
    results_mwb : dict
        Results from run_sweep (MWB detectors)
    results_ligo : dict
        Results from run_sweep_ligo
    alpha : float
        Gravitational coupling
    process_label : str
        Label for the physical process
    savepath : str, optional
        Path to save the figure
    """
    plt.rcParams.update({
        "text.usetex"         : True,
        "font.family"         : "serif",
        "font.serif"          : ["Computer Modern Roman"],
        "text.latex.preamble" : r"\usepackage{amsmath}"
    })

    fig, ax1 = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(top=0.88)

    # Colors for different detectors
    colors = {
        'ADMX-EFR': 'steelblue',
        'DMRadio-GUT': 'teal',
        'LIGO HF': 'purple',
    }
    
    linestyles = {
        'ADMX-EFR': '-',
        'DMRadio-GUT': '-',
        'LIGO HF': '--',
    }
    
    # Plot MWB detectors
    for det_key in ['det1', 'det2']:
        det_data = results_mwb[det_key]
        
        if len(det_data['M_combined']) == 0:
            continue
        
        M_plot = det_data['M_combined']
        d_plot = det_data['d_combined']
        
        # Sort by mass
        sort_idx = np.argsort(M_plot)
        M_plot = M_plot[sort_idx]
        d_plot = d_plot[sort_idx]
        
        # Remove duplicates
        _, unique_idx = np.unique(M_plot, return_index=True)
        M_plot = M_plot[unique_idx]
        d_plot = d_plot[unique_idx]
        
        ax1.loglog(M_plot, d_plot,
                   color=colors[det_data['name']],
                   linewidth=2.0,
                   linestyle=linestyles[det_data['name']],
                   label=det_data['name'])
        
        # Mark resonance
        if np.isfinite(det_data['d_res']) and np.isfinite(det_data['M_res']):
            ax1.axvline(det_data['M_res'], color=colors[det_data['name']], 
                       linewidth=1.0, linestyle=':', alpha=0.5, zorder=1)
            
            # Add annotation for resonance
            ax1.annotate(
                f'{det_data["name"]}\n$M = {det_data["M_res"]:.2e} M_\\odot$',
                xy=(det_data['M_res'], det_data['d_res']),
                xytext=(det_data['M_res'] * 0.5, det_data['d_res'] * 2),
                fontsize=8,
                color=colors[det_data['name']],
                arrowprops=dict(arrowstyle='->', color=colors[det_data['name']], lw=0.8),
                ha='center'
            )
    
    # Plot LIGO detector
    if len(results_ligo['M_combined']) > 0:
        M_plot = results_ligo['M_combined']
        d_plot = results_ligo['d_combined']
        
        sort_idx = np.argsort(M_plot)
        M_plot = M_plot[sort_idx]
        d_plot = d_plot[sort_idx]
        
        _, unique_idx = np.unique(M_plot, return_index=True)
        M_plot = M_plot[unique_idx]
        d_plot = d_plot[unique_idx]
        
        ax1.loglog(M_plot, d_plot,
                   color=colors['LIGO HF'],
                   linewidth=2.0,
                   linestyle=linestyles['LIGO HF'],
                   label='LIGO HF')
        
        # Mark LIGO resonance
        if np.isfinite(results_ligo['d_res']) and np.isfinite(results_ligo['M_res']):
            ax1.axvline(results_ligo['M_res'], color=colors['LIGO HF'], 
                       linewidth=1.0, linestyle=':', alpha=0.5, zorder=1)
            
            # Add annotation for LIGO resonance
            ax1.annotate(
                f'LIGO HF\n$M = {results_ligo["M_res"]:.2e} M_\\odot$',
                xy=(results_ligo['M_res'], results_ligo['d_res']),
                xytext=(results_ligo['M_res'] * 2, results_ligo['d_res'] * 0.3),
                fontsize=8,
                color=colors['LIGO HF'],
                arrowprops=dict(arrowstyle='->', color=colors['LIGO HF'], lw=0.8),
                ha='center'
            )

    # Add noise curves on right axis
    ax_noise = ax1.twinx()
    
    # Use ADMX-EFR data for frequency-to-mass mapping
    det1_data = results_mwb['det1']
    valid_coarse = np.isfinite(det1_data['d']) & np.isfinite(det1_data['M']) & (det1_data['d'] > 0)
    
    if valid_coarse.any():
        M_coarse = det1_data['M'][valid_coarse]
        f_coarse = det1_data['f'][valid_coarse]
        
        sort_idx = np.argsort(M_coarse)
        M_coarse_sorted = M_coarse[sort_idx]
        f_coarse_sorted = f_coarse[sort_idx]
        
        _, unique_idx = np.unique(M_coarse_sorted, return_index=True)
        M_coarse_unique = M_coarse_sorted[unique_idx]
        f_coarse_unique = f_coarse_sorted[unique_idx]
        
        if len(M_coarse_unique) >= 2:
            sort_f_idx = np.argsort(f_coarse_unique)
            f_sorted = f_coarse_unique[sort_f_idx]
            M_sorted = M_coarse_unique[sort_f_idx]
            
            _, f_unique_idx = np.unique(f_sorted, return_index=True)
            f_for_interp = f_sorted[f_unique_idx]
            M_for_interp = M_sorted[f_unique_idx]
            
            if len(f_for_interp) >= 2:
                f_to_M = interp1d(np.log10(f_for_interp), np.log10(M_for_interp),
                                 kind='linear', bounds_error=False, fill_value=np.nan)
                
                freqs_noise = np.logspace(2, 10, 5000)
                
                # MWB noise
                S_h_efr = noise_equivalent_strain_broadband(ADMX_EFR, freqs_noise)
                S_h_dmradio = noise_equivalent_strain_broadband(DMRADIO_GUT, freqs_noise)
                
                # LIGO noise
                S_h_ligo = get_ligo_noise_psd(freqs_noise, 'adv_ligo')
                
                # Convert to masses
                log_M_noise = f_to_M(np.log10(freqs_noise))
                valid_noise = np.isfinite(log_M_noise)
                
                if valid_noise.any():
                    M_noise = 10.0**log_M_noise[valid_noise]
                    freqs_valid = freqs_noise[valid_noise]
                    
                    # MWB noise
                    Sn_efr_valid = np.sqrt(S_h_efr[valid_noise])
                    Sn_dmradio_valid = np.sqrt(S_h_dmradio[valid_noise])
                    
                    # LIGO noise (where valid)
                    Sn_ligo_valid = np.sqrt(S_h_ligo[valid_noise])
                    
                    sort_m_idx = np.argsort(M_noise)
                    M_noise_sorted = M_noise[sort_m_idx]
                    Sn_efr_sorted = Sn_efr_valid[sort_m_idx]
                    Sn_dmradio_sorted = Sn_dmradio_valid[sort_m_idx]
                    Sn_ligo_sorted = Sn_ligo_valid[sort_m_idx]
                    
                    # Plot MWB noise
                    ax_noise.loglog(M_noise_sorted, Sn_efr_sorted,
                                   color='gray', linewidth=1.2, linestyle='--', alpha=0.7,
                                   label=r'ADMX-EFR $\sqrt{S_h^{\rm noise}}$')
                    ax_noise.loglog(M_noise_sorted, Sn_dmradio_sorted,
                                   color='darkgray', linewidth=1.2, linestyle=':', alpha=0.7,
                                   label=r'DMRadio-GUT $\sqrt{S_h^{\rm noise}}$')
                    
                    # Plot LIGO noise (only where finite)
                    valid_ligo_noise = np.isfinite(Sn_ligo_sorted)
                    if valid_ligo_noise.any():
                        ax_noise.loglog(M_noise_sorted[valid_ligo_noise], 
                                       Sn_ligo_sorted[valid_ligo_noise],
                                       color='purple', linewidth=1.2, linestyle='-.', alpha=0.7,
                                       label=r'LIGO HF $\sqrt{S_h^{\rm noise}}$')
    
    ax_noise.set_ylabel(r'$\left(S_h^{\rm noise}\right)^{1/2}\ [\mathrm{Hz}^{-1/2}]$',
                        fontsize=11, color='gray')
    ax_noise.tick_params(axis='y', labelcolor='gray')
    
    # Bottom axis labels
    ax1.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=13)
    ax1.set_ylabel(r'$d_{\rm max}\ [\mathrm{kpc}]$', fontsize=13)
    
    # Combine legends
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax_noise.get_legend_handles_labels()
    
    # Remove duplicate labels
    unique_labels = {}
    for line, label in zip(lines1 + lines2, labs1 + labs2):
        if label not in unique_labels:
            unique_labels[label] = line
    
    ax1.legend(unique_labels.values(), unique_labels.keys(),
              fontsize=9, loc='upper left', frameon=False)
    
    ax1.grid(True, alpha=0.3)
    
    # Title
    ax1.set_title(rf'Distance Reach: {process_label}, $\alpha = {alpha}$',
                 fontsize=12, pad=12)
    
    # Set limits
    ax1.set_xlim(1e-12, 1e-5)
    ax1.set_ylim(1e-10, 1e2)
    
    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f'[saved] {savepath}')
    
    plt.tight_layout()
    plt.show()
    return fig, ax1


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Common parameters
    alpha   = 0.01
    M_range = (1e-12, 1e4)
    
    # Transition parameters
    filepath = "2. Relativistic Superradiance Rate/Mathematica/SR_n2l1m1_at0.990_aMin0.010_aMax0.500_20260310.dat"
    transition = '3p 2p'
    process_label = r'$|422\rangle \to |322\rangle$ (transition)'
    
    # Create transition functions
    freq_func, h_func, tau_func = make_transition_funcs(
        alpha=alpha, transition=transition, filepath=filepath,
    )
    
    print("=" * 60)
    print("Computing MWB detector distance reach...")
    print("=" * 60)
    
    # Run MWB sweep
    results_mwb = run_sweep_mwb(
        freq_func=freq_func, h_func=h_func, tau_func=tau_func,
        M_range=M_range, rho_star=1.0,
    )
    
    print("\n" + "=" * 60)
    print("Computing LIGO detector distance reach...")
    print("=" * 60)
    
    # Run LIGO sweep
    # LIGO HF f_FSR = 37.5 kHz = 3.75e4 Hz
    results_ligo = run_sweep_ligo(
        freq_func=freq_func, h_func=h_func, tau_func=tau_func,
        M_range=M_range, rho_star=1.0, ligo_detector='adv_ligo',
        f_band=(3.75e4, 1e11)
    )
    
    print("\n" + "=" * 60)
    print("Plotting combined results...")
    print("=" * 60)
    
    # Print peak distances
    print("\nPeak Distance Reaches:")
    print("-" * 40)
    
    # MWB detectors
    for det_key in ['det1', 'det2']:
        det_data = results_mwb[det_key]
        if np.isfinite(det_data['d_res']):
            print(f"{det_data['name']}: {det_data['d_res']:.2e} kpc at M = {det_data['M_res']:.2e} Msun")
    
    # LIGO detector
    if np.isfinite(results_ligo['d_res']):
        print(f"LIGO HF: {results_ligo['d_res']:.2e} kpc at M = {results_ligo['M_res']:.2e} Msun")
    
    # Plot combined results
    plot_combined_reach(
        results_mwb, results_ligo, alpha, process_label,
        savepath='4. Detector Distance Reach/combined_distance_reach_ligo_mwb.pdf'
    )