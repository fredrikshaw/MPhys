"""
AlphaReach.py
=============

Computes and plots the maximum detectable distance d_max(alpha) for
superradiance GW sources, using the peak-strain output files produced by
run_alpha_sweep() in superradiance_simulation.py.

For a fixed BH mass M_BH and spin ã₀, one .dat file is produced per alpha
by the simulation sweep.  This script reads the entire alpha-sweep data
directory, converts the peak GW strain at 1 kpc into a detector distance
reach for each detector, and overlays the PBH merger-rate reach as a
horizontal line (constant for fixed mass).

Detectors
---------
- ADMX-EFR         (Magnetic Weber Bar)
- DMRadio-GUT      (Magnetic Weber Bar)
- LIGO HF          (IFO, Schnabel & Korobko extrapolation)

Usage
-----
Edit the USER PARAMETERS block at the bottom and run:

    python AlphaReach.py

Physics
-------
Each simulation .dat file stores h_peak at 1 kpc and a GW frequency.
The conversion to detector reach is:

    h_unit   = h_peak_1kpc * KPC_TO_M          [strain at 1 m]
    tau_obs  = min(fwhm_yr * YEAR_S, ONE_YEAR)  [capped at 1 yr]
    d_max(m) = h_unit * sqrt(tau_obs / S_h) / rho_star
    d_max(kpc) = d_max(m) / KPC_TO_M

where S_h is the detector noise PSD at the emission frequency [Hz^-1].
The ring-up threshold is respected: if tau < t_ring, the signal is
undetectable by that detector.

The best (largest d_max) level is selected per alpha point per detector,
so the plotted curve is the envelope over all SR levels.
"""

import os
import re
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Import all detector machinery from GWDetectors.py
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "4. Detector Distance Reach"))
from GWDetectors import (
    G_N_NAT, EV_TO_SI, INV_EV_TO_M, KPC_TO_M,
    MagneticWeberBar, ADMX_EFR, DMRADIO_GUT, MWB_DETECTORS,
    mwb_noise_psd,
    IFOConfig, IFO_DETECTORS,
    ifo_noise_psd,
)

# ─────────────────────────────────────────────────────────────────────────────
# Resolve merger-rate module paths (same directory-walking as DetectorReach.py)
# ─────────────────────────────────────────────────────────────────────────────

current_dir = Path(__file__).resolve().parent
sem2_dir    = None
for p in current_dir.parents:
    if p.name == "Sem 2":
        sem2_dir = p
        break
if sem2_dir is None:
    sem2_dir = current_dir.parent

merger_dir = sem2_dir / "6. PBH spin and mass distribution"
if not merger_dir.exists():
    for p in current_dir.parents:
        candidate = p / "6. PBH spin and mass distribution"
        if candidate.exists():
            merger_dir = candidate
            break
sys.path.append(str(merger_dir.resolve()))

from MergerRate import (
    convert_rate_volume_unit,
    integrate_rate_over_square_grid,
    total_rate_above_spin_threshold,
)

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────

ONE_YEAR_S = 365.25 * 24.0 * 3600.0    # [s]
YEAR_S     = ONE_YEAR_S                 # same value, named for clarity in conversions


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_alpha_sweep_file(filepath):
    """
    Load a single peak-strain .dat file produced by save_peak_tables().

    The file is tab-delimited with a header row and one row per SR level.
    All available columns are imported; t_peak_yr is stored but not
    currently used in the reach calculation.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    list of dicts, one per level, with keys:
        label        : str   — level label, e.g. '|211⟩'
        peak_strain  : float — peak GW strain at 1 kpc  [dimensionless]
        fwhm_yr      : float — FWHM signal duration  [yr]
        t_peak_yr    : float — time of peak  [yr]  (imported; not used yet)
        frequency_hz : float — GW emission frequency  [Hz]
    """
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append({
                'label'       : row['label'],
                'peak_strain' : float(row['peak_strain']),
                'fwhm_yr'     : float(row['fwhm_yr']),
                't_peak_yr'   : float(row['t_peak_yr']),
                'frequency_hz': float(row['frequency_hz']),
            })
    return rows


def parse_alpha_from_filename(filename):
    """
    Extract the alpha value encoded in a save_peak_tables() filename.

    The naming convention is:
        peaks_{process}_{mass_str}_{alpha_str}_{spin_str}.dat

    where alpha_str is of the form 'alpha<value>', with the decimal point
    produced by save_peak_tables() or replaced by an underscore on some
    operating systems.  Both forms are handled.

    Examples
    --------
    peaks_annihilation_M1e-6_alpha0.4_a0.65.dat   → 0.4
    peaks_annihilation_M1e-6_alpha0_4_a0_65.dat   → 0.4
    peaks_annihilation_M1e-6_alpha0.001_a0.65.dat → 0.001
    peaks_annihilation_M1e-6_alpha1.5_a0.65.dat   → 1.5

    Raises
    ------
    ValueError if the alpha token cannot be parsed.
    """
    # Capture everything between 'alpha' and the next '_a' spin token or '.dat'
    m = re.search(r'alpha([\d._]+?)(?=_a[\d]|\.dat)', filename)
    if m:
        raw = m.group(1)
        # Normalise: replace underscores used as decimal points with dots,
        # then collapse any runs of dots to a single dot.
        raw = raw.replace('_', '.')
        raw = re.sub(r'\.{2,}', '.', raw).strip('.')
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    raise ValueError(f"Could not parse alpha from filename: {filename!r}")


def load_all_sweep_files(data_dir, process='annihilation'):
    """
    Load all peak-strain files of the requested process type from data_dir.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the .dat files from run_alpha_sweep().
    process  : str
        'annihilation' or 'transitions' — selects which file prefix to glob.

    Returns
    -------
    sweep : list of dicts, each containing:
        'alpha'  : float — the alpha value for this file
        'levels' : list  — output of load_alpha_sweep_file()
    Sorted by ascending alpha.
    """
    data_dir = Path(data_dir)
    pattern  = f'peaks_{process}_*.dat'
    files    = sorted(data_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No '{pattern}' files found in '{data_dir}'.\n"
            f"  Check DATA_DIR and PROCESS in the USER PARAMETERS block."
        )

    sweep = []
    for fp in files:
        try:
            alpha  = parse_alpha_from_filename(fp.name)
            levels = load_alpha_sweep_file(fp)
            sweep.append({'alpha': alpha, 'levels': levels})
            print(f"  Loaded α = {alpha:.5g}  ({len(levels)} levels)  [{fp.name}]")
        except Exception as exc:
            print(f"  [WARN] Skipping {fp.name}: {exc}")

    sweep.sort(key=lambda x: x['alpha'])
    print(f"Total: {len(sweep)} alpha points loaded.\n")
    return sweep


# ═════════════════════════════════════════════════════════════════════════════
# Detector noise helpers  (mirrors DetectorReach.py)
# ═════════════════════════════════════════════════════════════════════════════

def _get_noise(det, f_t):
    """
    Return S_h_noise [Hz^-1] at frequency f_t for a MWB or IFO detector.

    Parameters
    ----------
    det  : MagneticWeberBar instance  or  str IFO key (e.g. 'adv_ligo')
    f_t  : float — frequency [Hz]
    """
    if isinstance(det, MagneticWeberBar):
        return mwb_noise_psd(det, np.array([f_t]))[0]
    elif isinstance(det, str):
        return ifo_noise_psd(np.array([f_t]), detector_key=det)[0]
    else:
        raise TypeError(f"det must be MagneticWeberBar or str, got {type(det)}")


def _get_ring_up_time(det):
    """Return ring-up time [s] for a MWB or IFO detector."""
    if isinstance(det, MagneticWeberBar):
        return det.ring_up_time()
    elif isinstance(det, str):
        return IFO_DETECTORS[det].ring_up_time()
    else:
        raise TypeError(f"det must be MagneticWeberBar or str, got {type(det)}")


# ═════════════════════════════════════════════════════════════════════════════
# d_max computation from simulation peak data
# ═════════════════════════════════════════════════════════════════════════════

def compute_d_max_from_peak(peak_strain_1kpc, frequency_hz, fwhm_yr,
                             det, rho_star=1.0):
    """
    Compute the maximum detectable distance for a single SR level and detector.

    Parameters
    ----------
    peak_strain_1kpc : float   Peak GW strain at 1 kpc (from simulation output).
    frequency_hz     : float   GW emission frequency [Hz].
    fwhm_yr          : float   Signal duration (FWHM from simulation) [yr].
    det              : MagneticWeberBar | str  Detector instance or IFO key.
    rho_star         : float   SNR threshold (default 1).

    Returns
    -------
    d_max_kpc : float or nan
        nan if the detector cannot observe this signal (frequency out of band,
        duration below ring-up threshold, non-finite noise PSD, etc.).

    Notes
    -----
    h_unit   = peak_strain_1kpc * KPC_TO_M      [strain at 1 m, scales as 1/r]
    tau_obs  = min(fwhm_yr * YEAR_S, ONE_YEAR_S)
    d_max(m) = h_unit * sqrt(tau_obs / S_h) / rho_star
    """
    if not (np.isfinite(peak_strain_1kpc) and peak_strain_1kpc > 0):
        return np.nan
    if not (np.isfinite(frequency_hz) and frequency_hz > 0):
        return np.nan
    if not (np.isfinite(fwhm_yr) and fwhm_yr > 0):
        return np.nan

    # Strain at 1 m (strain ∝ 1/r, so multiply by kpc in metres)
    h_unit = peak_strain_1kpc * KPC_TO_M

    # Signal duration with 1-year cap; enforce ring-up threshold
    tau_val = fwhm_yr * YEAR_S                  # [s]
    try:
        t_ring = _get_ring_up_time(det)
    except Exception:
        t_ring = 0.0
    if tau_val < t_ring:
        return np.nan

    tau_obs = min(tau_val, ONE_YEAR_S)          # [s]

    # Noise PSD at the signal frequency
    try:
        S_h = _get_noise(det, frequency_hz)
    except Exception:
        return np.nan
    if not (np.isfinite(S_h) and S_h > 0):
        return np.nan

    d_max_m   = h_unit * np.sqrt(tau_obs / S_h) / rho_star
    d_max_kpc = d_max_m / KPC_TO_M

    return d_max_kpc if (np.isfinite(d_max_kpc) and d_max_kpc > 0) else np.nan


def best_d_max_for_alpha(levels, det, rho_star=1.0):
    """
    For a single alpha point, return the largest d_max across all SR levels.

    This gives the detection envelope: the strongest level drives the reach.

    Parameters
    ----------
    levels   : list of dicts — output of load_alpha_sweep_file()
    det      : MagneticWeberBar | str
    rho_star : float

    Returns
    -------
    d_max_kpc : float or nan
    """
    best = np.nan
    for lev in levels:
        d = compute_d_max_from_peak(
            lev['peak_strain'],
            lev['frequency_hz'],
            lev['fwhm_yr'],
            det,
            rho_star,
        )
        if np.isfinite(d) and (not np.isfinite(best) or d > best):
            best = d
    return best


# ═════════════════════════════════════════════════════════════════════════════
# Main reach sweep over all alpha points and detectors
# ═════════════════════════════════════════════════════════════════════════════

def run_alpha_reach_sweep(sweep_data, rho_star=1.0, include_ligo=True):
    """
    Compute d_max(alpha) for every detector.

    Parameters
    ----------
    sweep_data   : list — output of load_all_sweep_files()
    rho_star     : float — SNR threshold
    include_ligo : bool  — include IFO detectors alongside MWB

    Returns
    -------
    alphas : np.ndarray
        Sorted array of alpha values.
    reach : dict
        Keys are detector display names; values are np.ndarray of d_max_kpc
        (nan where the source is undetectable).
    """
    alphas = np.array([pt['alpha'] for pt in sweep_data])

    # Build ordered list of (display_name, det_key_or_instance)
    detectors = []
    for det in MWB_DETECTORS:
        detectors.append((det.name, det))
    if include_ligo:
        for key, ifo in IFO_DETECTORS.items():
            detectors.append((ifo.name, key))

    reach = {name: np.full(len(alphas), np.nan) for name, _ in detectors}

    print(f"{'α':>10}  " + "  ".join(f"{name:>14}" for name, _ in detectors))
    print("-" * (12 + 16 * len(detectors)))

    for i, pt in enumerate(sweep_data):
        alpha  = pt['alpha']
        levels = pt['levels']
        row    = f"{alpha:>10.5g}  "
        for det_name, det in detectors:
            d = best_d_max_for_alpha(levels, det, rho_star)
            reach[det_name][i] = d
            tag = f"{d:.2e}" if np.isfinite(d) else "     ---"
            row += f"{tag:>14}  "
        print(row + "kpc")

    print()
    return alphas, reach


# ═════════════════════════════════════════════════════════════════════════════
# Merger rate reach for a fixed BH mass  (horizontal line on the plot)
# ═════════════════════════════════════════════════════════════════════════════

def _rate_density_to_distance_kpc(rate_density_kpc):
    """
    Convert a uniform event-rate density [yr^-1 kpc^-3] to a
    1-event-per-year radius [kpc].

    Assumes sources are uniformly distributed in a sphere of radius d:
        R = (4π/3) d³ × rate_density  →  d = (3 / (4π × rate_density))^(1/3)
    """
    rate = float(rate_density_kpc)
    if rate > 0:
        return (3.0 / (4.0 * np.pi * rate)) ** (1.0 / 3.0)
    return np.nan


def compute_merger_reach_fixed_mass(
    M_BH_solar,
    a_star_threshold=0.6,
    spin_model='matched',
    mass_sigma=0.5,
    fpbh=1.0,
    t_over_t0=1.0,
    rate_grid_points=90,
):
    """
    Compute the 1-event/year PBH merger reach for a fixed post-merger mass.

    Because M_BH is fixed, this returns a single distance value (not a curve),
    which becomes a horizontal line on the alpha plot.

    The component mass is taken as M_BH_solar / 2, consistent with the
    convention in DetectorReach.py (post-merger mass ≈ sum of components).

    Parameters
    ----------
    M_BH_solar       : float — fixed BH mass [M_sun]
    a_star_threshold : float — spin threshold for the spin-cut version
    spin_model       : str   — spin distribution model (see MergerRate.py)
    mass_sigma       : float — log-normal width of the mass distribution
    fpbh             : float — PBH dark matter fraction
    t_over_t0        : float — time normalisation (1 = today)
    rate_grid_points : int   — grid resolution for the rate integral

    Returns
    -------
    reach_spin_kpc : float or nan — reach with spin-threshold applied
    reach_all_kpc  : float or nan — reach with no spin cut
    """
    component_mass = 0.5 * M_BH_solar
    m_min = component_mass * np.exp(-5.0 * mass_sigma)
    m_max = component_mass * np.exp(+5.0 * mass_sigma)

    reach_spin = np.nan
    reach_all  = np.nan

    try:
        rate_spin = total_rate_above_spin_threshold(
            m_min=m_min, m_max=m_max, n_points=rate_grid_points,
            m_c=component_mass, sigma=mass_sigma,
            a_star_threshold=a_star_threshold, spin_model=spin_model,
            fpbh=fpbh, t_over_t0=t_over_t0, volume_unit='kpc',
        )
        reach_spin = _rate_density_to_distance_kpc(rate_spin)
        print(f"  Merger reach (a* ≥ {a_star_threshold}):  {reach_spin:.3e} kpc")
    except Exception as exc:
        print(f"  [WARN] Merger reach (spin cut) failed: {exc}")

    try:
        rate_all_gpc = integrate_rate_over_square_grid(
            m_min=m_min, m_max=m_max, n_points=rate_grid_points,
            m_c=component_mass, sigma=mass_sigma,
            fpbh=fpbh, t_over_t0=t_over_t0, nu_threshold=None,
        )
        rate_all_kpc = convert_rate_volume_unit(rate_all_gpc,
                                                from_unit='gpc',
                                                to_unit='kpc')
        reach_all = _rate_density_to_distance_kpc(rate_all_kpc)
        print(f"  Merger reach (no spin cut):        {reach_all:.3e} kpc")
    except Exception as exc:
        print(f"  [WARN] Merger reach (all mergers) failed: {exc}")

    return reach_spin, reach_all


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

# Colour and line style per detector name
_DET_STYLE = {
    'ADMX-EFR'   : {'color': 'steelblue', 'ls': '-'},
    'DMRadio-GUT': {'color': 'teal',       'ls': '-'},
    'LIGO HF'    : {'color': 'black',      'ls': '--'},
}


def plot_alpha_reach(
    alphas, reach, M_BH_solar, process_label,
    show_merger_reach=True,
    merger_spin_threshold=0.6,
    merger_spin_model='matched',
    merger_mass_sigma=0.5,
    merger_fpbh=1.0,
    merger_t_over_t0=1.0,
    merger_rate_grid_points=90,
    xlim=None,
    ylim=None,
    savepath=None,
):
    """
    Plot d_max vs alpha for all detectors and (optionally) the PBH merger line.

    Parameters
    ----------
    alphas        : np.ndarray — alpha grid from run_alpha_reach_sweep()
    reach         : dict       — {detector_name: d_max array}
    M_BH_solar    : float      — fixed BH mass (shown in title)
    process_label : str        — LaTeX description of the SR process
    show_merger_reach : bool   — overlay horizontal merger rate line
    merger_*      : various    — merger rate parameters (see compute_merger_reach_fixed_mass)
    xlim, ylim    : tuple or None — axis limits; auto if None
    savepath      : str or None  — file path to save; not saved if None

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    plt.rcParams.update({
        "text.usetex"        : True,
        "font.family"        : "serif",
        "font.serif"         : ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.subplots_adjust(top=0.82)

    # ── Detector reach curves ─────────────────────────────────────────────────
    for det_name, d_arr in reach.items():
        style = _DET_STYLE.get(det_name, {'color': 'gray', 'ls': '-'})
        mask  = np.isfinite(d_arr) & (d_arr > 0)
        if not mask.any():
            print(f"  [INFO] {det_name}: no finite d_max values to plot.")
            continue
        ax.loglog(alphas[mask], d_arr[mask],
                  color=style['color'],
                  ls=style['ls'],
                  linewidth=2.0,
                  label=det_name)

    # ── PBH merger reach — horizontal line (constant mass) ───────────────────
    if show_merger_reach:
        print("Computing PBH merger reach for fixed mass...")
        reach_spin, reach_all = compute_merger_reach_fixed_mass(
            M_BH_solar,
            a_star_threshold=merger_spin_threshold,
            spin_model=merger_spin_model,
            mass_sigma=merger_mass_sigma,
            fpbh=merger_fpbh,
            t_over_t0=merger_t_over_t0,
            rate_grid_points=merger_rate_grid_points,
        )

        if np.isfinite(reach_spin):
            ax.axhline(
                reach_spin,
                color='firebrick', linewidth=2.0, linestyle='-',
                label=rf'PBH mergers, $a_* \geq {merger_spin_threshold:.2f}$',
            )
        if np.isfinite(reach_all):
            ax.axhline(
                reach_all,
                color='firebrick', linewidth=1.5, linestyle=':',
                label='PBH mergers, no spin cut',
            )

    # ── Axis labels and limits ────────────────────────────────────────────────
    ax.set_xlabel(r'$\alpha = G M_{\rm BH} \mu / \hbar c$', fontsize=13)
    ax.set_ylabel(r'$d_{\rm max}\ [\mathrm{kpc}]$',         fontsize=13)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.legend(fontsize=9, loc='best', frameon=False)
    ax.grid(True, which='both', alpha=0.25, linestyle='--')

    mass_str = f'{M_BH_solar:.2g}'
    ax.set_title(
        process_label + '\n'
        + rf'$M_{{\rm BH}} = {mass_str}\,M_\odot$',
        fontsize=10, pad=6,
    )

    plt.tight_layout()

    if savepath is not None:
        save_dir = Path(savepath).parent
        if str(save_dir) not in ('', '.'):
            save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f'[saved] {savepath}')

    plt.show()
    return fig, ax


# ═════════════════════════════════════════════════════════════════════════════
# Main — edit ONLY this block
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Fixed physical parameters ─────────────────────────────────────────────
    M_BH_SOLAR = 1e-6           # BH mass [solar masses] — fixed across the sweep
    PROCESS    = 'annihilation' # 'annihilation' or 'transitions'

    # ── Data directory  (output of run_alpha_sweep in superradiance_simulation.py)
    DATA_DIR = 'Sem 2/8. Numerical Simulations/Data/alpha_sweep'

    # ── Detection threshold ───────────────────────────────────────────────────
    RHO_STAR = 1.0              # SNR threshold ρ*

    # ── Merger rate parameters ────────────────────────────────────────────────
    SHOW_MERGER_REACH       = True
    MERGER_SPIN_THRESHOLD   = 0.6
    MERGER_MASS_SIGMA       = 0.5
    MERGER_FPBH             = 0.1
    MERGER_T_OVER_T0        = 1.0
    MERGER_RATE_GRID_POINTS = 90

    # ── Plot parameters ───────────────────────────────────────────────────────
    SAVEDIR  = 'Sem 2/8. Numerical Simulations/Plots'
    SAVEPATH = f'{SAVEDIR}/reach_vs_alpha_{PROCESS}.pdf'
    XLIM     = None     # e.g. (0.001, 1.5); set None for auto
    YLIM     = None     # e.g. (1e-3, 1e6);  set None for auto

    process_label = (
        r'Axion cloud annihilation ($m = l$ levels)'
        if PROCESS == 'annihilation'
        else r'Axion cloud transitions ($m = l$ levels)'
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"Alpha reach sweep  —  {PROCESS}  —  M_BH = {M_BH_SOLAR} M_sun")
    print("=" * 65)
    print(f"\nLoading alpha-sweep data from '{DATA_DIR}'...")
    sweep_data = load_all_sweep_files(DATA_DIR, process=PROCESS)

    # ── Compute reach for all detectors ───────────────────────────────────────
    print("Computing d_max(alpha) for all detectors...\n")
    alphas, reach = run_alpha_reach_sweep(
        sweep_data, rho_star=RHO_STAR, include_ligo=True,
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("Plotting...")
    plot_alpha_reach(
        alphas, reach, M_BH_SOLAR, process_label,
        show_merger_reach       = SHOW_MERGER_REACH,
        merger_spin_threshold   = MERGER_SPIN_THRESHOLD,
        merger_spin_model       = 'matched',
        merger_mass_sigma       = MERGER_MASS_SIGMA,
        merger_fpbh             = MERGER_FPBH,
        merger_t_over_t0        = MERGER_T_OVER_T0,
        merger_rate_grid_points = MERGER_RATE_GRID_POINTS,
        xlim     = XLIM,
        ylim     = YLIM,
        savepath = SAVEPATH,
    )