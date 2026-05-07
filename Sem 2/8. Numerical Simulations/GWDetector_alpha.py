"""
GWDetector_alpha.py
===================

Computes and plots the maximum detectable distance d_max(alpha) for
superradiance GW sources, using the peak-strain output files produced by
run_alpha_sweep() in superradiance_simulation.py.

For a fixed BH mass M_BH and spin a_star, one .dat file is produced per
alpha by the simulation sweep.  This script reads the entire alpha-sweep
data directory, converts the peak GW strain at 1 kpc into a detector
distance reach for each detector, and overlays the PBH merger-rate reach
as a horizontal line (constant for fixed mass).

Detectors
---------
- ADMX-EFR         (Magnetic Weber Bar)
- DMRadio-GUT      (Magnetic Weber Bar)
- LIGO HF          (IFO, Schnabel & Korobko extrapolation)

Usage
-----
Edit the USER PARAMETERS block at the bottom and run:

    python GWDetector_alpha.py

Physics
-------
Each simulation .dat file stores h_peak at 1 kpc and a GW frequency.
The conversion to detector reach is:

    h_unit     = h_peak_1kpc * KPC_TO_M            [strain at 1 m]
    tau_obs    = min(fwhm_yr * YEAR_S, ONE_YEAR_S)  [capped at 1 yr]
    d_max [m]  = h_unit * sqrt(tau_obs / S_h) / rho_star
    d_max [kpc]= d_max [m] / KPC_TO_M

where S_h is the detector noise PSD at the emission frequency [Hz^-1].

Filters applied per level before computing d_max
-------------------------------------------------
1. Superradiance condition:  alpha < m * a_star / (2*(1+sqrt(1-a_star^2)))
   Levels that violate this are not SR-active and are skipped.
2. Cosmological time cut:  t_peak_yr <= T_PEAK_MAX_YR (default 1e8 yr)
   Signals that peak after this are excluded.
3. Frequency gating:  _get_noise() returns NaN for frequencies outside
   each detector's sensitive band; compute_d_max_from_peak() propagates
   this to a NaN reach, so out-of-band signals never appear on the plot.
"""

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
# Resolve merger-rate module paths
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
YEAR_S     = ONE_YEAR_S

# SI constants used for the axion mass top axis
# mu [eV] = alpha / r_g [eV^-1],  r_g [eV^-1] = G M / c^2 / (hbar*c/eV)
_G_SI        = 6.674e-11    # m^3 kg^-1 s^-2
_C_SI        = 2.998e8      # m s^-1
_M_SUN_KG    = 1.989e30     # kg
_HBAR_C_EV_M = 1.973e-7     # hbar*c in eV*m

# Maximum t_peak: signals peaking later than this are excluded
T_PEAK_MAX_YR = 1e9         # [yr]


# ═════════════════════════════════════════════════════════════════════════════
# Filename parsing
# ═════════════════════════════════════════════════════════════════════════════

def parse_alpha_from_filename(filename):
    """
    Extract the alpha value from a save_peak_tables() filename.

    Naming convention:
        peaks_{process}_M{mass}_alpha{alpha}_a{spin}.dat

    Handles both dot and underscore as decimal separator.

    Examples
    --------
    peaks_annihilation_M1e-6_alpha0.4_a0.65.dat   -> 0.4
    peaks_transitions_M1e-6_alpha0.001_a0.65.dat  -> 0.001
    peaks_annihilation_M1e-6_alpha0_4_a0_65.dat   -> 0.4
    """
    stem = filename
    if stem.endswith('.dat'):
        stem = stem[:-4]

    # Find 'alpha' then take up to the next '_a' (spin token)
    idx = stem.find('alpha')
    if idx == -1:
        raise ValueError(f"Could not parse alpha from filename: {filename!r}")

    raw = stem[idx + 5:]            # everything after 'alpha'
    end = raw.find('_a')
    if end != -1:
        raw = raw[:end]

    raw = raw.replace('_', '.').strip('.')
    raw = re.sub(r'\.{2,}', '.', raw)

    try:
        value = float(raw)
        if value > 0:
            return value
    except ValueError:
        pass
    raise ValueError(f"Could not parse alpha from filename: {filename!r}")


def parse_mass_from_filename(filename):
    """
    Extract the BH mass from a save_peak_tables() filename.

    Examples
    --------
    peaks_annihilation_M1e-6_alpha0.4_a0.65.dat  -> 1e-6
    peaks_annihilation_M1e-11_alpha0.1_a0.65.dat -> 1e-11
    """
    stem = filename
    if stem.endswith('.dat'):
        stem = stem[:-4]

    # Find '_M' then take up to '_alpha'
    idx = stem.find('_M')
    if idx == -1:
        raise ValueError(f"Could not parse mass from filename: {filename!r}")

    raw = stem[idx + 2:]
    end = raw.find('_alpha')
    if end != -1:
        raw = raw[:end]

    raw = raw.replace('_', '.').strip('.')
    raw = re.sub(r'\.{2,}', '.', raw)

    try:
        value = float(raw)
        if value > 0:
            return value
    except ValueError:
        pass
    raise ValueError(f"Could not parse mass from filename: {filename!r}")


def parse_spin_from_filename(filename):
    """
    Extract the initial BH spin a_star from a save_peak_tables() filename.

    The spin is the last token in the stem, after the final '_a'.

    Examples
    --------
    peaks_transitions_M1e-6_alpha0.001_a0.65.dat -> 0.65
    peaks_annihilation_M1e-6_alpha0_4_a0_65.dat  -> 0.65
    """
    stem = filename
    if stem.endswith('.dat'):
        stem = stem[:-4]

    # Take everything after the LAST '_a' in the stem
    idx = stem.rfind('_a')
    if idx == -1:
        raise ValueError(f"Could not parse spin from filename: {filename!r}")

    raw = stem[idx + 2:]
    raw = raw.replace('_', '.').strip('.')
    raw = re.sub(r'\.{2,}', '.', raw)

    try:
        value = float(raw)
        if 0.0 < value <= 1.0:
            return value
    except ValueError:
        pass
    raise ValueError(f"Could not parse spin from filename: {filename!r}")


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_alpha_sweep_file(filepath):
    """
    Load a single peak-strain .dat file produced by save_peak_tables().

    Tab-delimited with a header row; one row per SR level.
    All columns are imported including t_peak_yr (stored for future use).

    Returns
    -------
    list of dicts with keys:
        label        : str   -- level label, e.g. '|211>'
        peak_strain  : float -- peak GW strain at 1 kpc  [dimensionless]
        fwhm_yr      : float -- FWHM signal duration  [yr]
        t_peak_yr    : float -- time of peak  [yr]
        frequency_hz : float -- GW emission frequency  [Hz]
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


def load_all_sweep_files(data_dir, process='annihilation', mass_filter=None):
    """
    Load all peak-strain files for the requested process from data_dir,
    optionally restricting to files whose encoded BH mass matches mass_filter.

    Parameters
    ----------
    data_dir    : str or Path
    process     : str -- 'annihilation' or 'transitions'
    mass_filter : float or None -- if set, only files with a matching mass
                  token (within 0.1%) are loaded

    Returns
    -------
    sweep : list of dicts (sorted by ascending alpha), each with:
        'alpha'  : float
        'a_star' : float
        'levels' : list of level dicts from load_alpha_sweep_file()
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
        # ── Mass filter ───────────────────────────────────────────────────────
        if mass_filter is not None:
            try:
                file_mass = parse_mass_from_filename(fp.name)
            except ValueError as exc:
                print(f"  [WARN] Skipping {fp.name}: {exc}")
                continue
            if not np.isclose(file_mass, mass_filter, rtol=1e-3):
                continue

        # ── Parse metadata ────────────────────────────────────────────────────
        try:
            alpha  = parse_alpha_from_filename(fp.name)
            a_star = parse_spin_from_filename(fp.name)
            levels = load_alpha_sweep_file(fp)
        except Exception as exc:
            print(f"  [WARN] Skipping {fp.name}: {exc}")
            continue

        sweep.append({'alpha': alpha, 'a_star': a_star, 'levels': levels})
        print(f"  Loaded alpha = {alpha:.5g}  a* = {a_star}  "
              f"({len(levels)} levels)  [{fp.name}]")

    sweep.sort(key=lambda x: x['alpha'])
    print(f"Total: {len(sweep)} alpha points loaded.\n")
    return sweep


# ═════════════════════════════════════════════════════════════════════════════
# Superradiance condition helpers
# ═════════════════════════════════════════════════════════════════════════════

def _parse_m_from_label(label):
    """
    Extract the azimuthal quantum number m from a level label string.

    Takes the third digit of the first |nlm> state in the label, which is
    the excited state for transitions and the only state for annihilation.

    Examples:  '|211>' -> 1,  '|322>' -> 2,  '|644>->|544>' -> 4
    """
    match = re.search(r'\|(\d)(\d)(\d)', label)
    if match:
        return int(match.group(3))
    return None


def _is_superradiant(alpha, a_star, m):
    """
    Return True if the level satisfies the superradiance condition.

    The condition omega_nlm < m * Omega_H reduces (the binding correction
    alpha^2/n^2 cancels from both sides) to:

        alpha < m * a_star / (2 * (1 + sqrt(1 - a_star^2)))

    Parameters
    ----------
    alpha  : float -- gravitational fine-structure constant
    a_star : float -- dimensionless BH spin
    m      : int   -- azimuthal quantum number
    """
    if a_star <= 0.0 or m <= 0:
        return False
    Omega_H_factor = a_star / (2.0 * (1.0 + np.sqrt(max(0.0, 1.0 - a_star**2))))
    return alpha < m * Omega_H_factor


# ═════════════════════════════════════════════════════════════════════════════
# Detector noise helpers
# ═════════════════════════════════════════════════════════════════════════════

def _get_noise(det, f_t):
    """
    Return S_h_noise [Hz^-1] at frequency f_t.

    Returns NaN for frequencies outside the detector's sensitive band,
    which causes compute_d_max_from_peak() to return NaN -- those signals
    are automatically excluded from the plot.
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
    peak_strain_1kpc : float -- peak GW strain at 1 kpc
    frequency_hz     : float -- GW emission frequency [Hz]
    fwhm_yr          : float -- signal duration (FWHM) [yr]
    det              : MagneticWeberBar | str
    rho_star         : float -- SNR threshold

    Returns
    -------
    d_max_kpc : float or nan

    Notes
    -----
    h_unit   = peak_strain_1kpc * KPC_TO_M      [strain at 1 m]
    tau_obs  = min(fwhm_yr * YEAR_S, ONE_YEAR_S)
    d_max(m) = h_unit * sqrt(tau_obs / S_h) / rho_star

    Out-of-band frequencies return NaN via _get_noise(); the ring-up
    threshold is enforced before any distance is computed.
    """
    if not (np.isfinite(peak_strain_1kpc) and peak_strain_1kpc > 1e-50):
        return np.nan
    if not (np.isfinite(frequency_hz) and frequency_hz > 0):
        return np.nan
    if not (np.isfinite(fwhm_yr) and fwhm_yr > 0):
        return np.nan

    h_unit = peak_strain_1kpc * KPC_TO_M       # strain at 1 m

    tau_val = fwhm_yr * YEAR_S                  # [s]
    try:
        t_ring = _get_ring_up_time(det)
    except Exception:
        t_ring = 0.0
    if tau_val < t_ring:
        return np.nan

    tau_obs = min(tau_val, ONE_YEAR_S)

    try:
        S_h = _get_noise(det, frequency_hz)
    except Exception:
        return np.nan
    if not (np.isfinite(S_h) and S_h > 0):
        return np.nan

    d_max_m   = h_unit * np.sqrt(tau_obs / S_h) / rho_star
    d_max_kpc = d_max_m / KPC_TO_M

    return d_max_kpc if (np.isfinite(d_max_kpc) and d_max_kpc > 0) else np.nan


def best_d_max_for_alpha(levels, det, rho_star=1.0, level_label=None,
                          alpha=None, a_star=None,
                          t_peak_max_yr=T_PEAK_MAX_YR):
    """
    For a single alpha point, return the largest d_max across SR levels.

    Parameters
    ----------
    levels        : list of dicts from load_alpha_sweep_file()
    det           : MagneticWeberBar | str
    rho_star      : float
    level_label   : str or None -- if set, only that level is considered
    alpha         : float or None -- used for SR condition check
    a_star        : float or None -- used for SR condition check
    t_peak_max_yr : float -- signals peaking after this [yr] are excluded

    Returns
    -------
    d_max_kpc : float or nan
    """
    if level_label is not None:
        levels = [lev for lev in levels if lev['label'] == level_label]
        if not levels:
            return np.nan

    best = np.nan
    for lev in levels:

        # ── Superradiance condition ───────────────────────────────────────────
        if alpha is not None and a_star is not None:
            m = _parse_m_from_label(lev['label'])
            if m is not None and not _is_superradiant(alpha, a_star, m):
                continue

        # ── Cosmological time cutoff ──────────────────────────────────────────
        if lev['t_peak_yr'] > t_peak_max_yr:
            continue

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

def run_alpha_reach_sweep(sweep_data, rho_star=1.0, include_ligo=True,
                          level_label=None):
    """
    Compute d_max(alpha) for every detector.

    Parameters
    ----------
    sweep_data   : list -- output of load_all_sweep_files()
    rho_star     : float -- SNR threshold
    include_ligo : bool  -- include IFO detectors alongside MWB
    level_label  : str or None -- restrict to a single SR level

    Returns
    -------
    alphas : np.ndarray -- sorted alpha values
    reach  : dict -- {detector_name: np.ndarray of d_max_kpc}
    """
    alphas = np.array([pt['alpha'] for pt in sweep_data])

    detectors = []
    for det in MWB_DETECTORS:
        detectors.append((det.name, det))
    if include_ligo:
        for key, ifo in IFO_DETECTORS.items():
            detectors.append((ifo.name, key))

    reach = {name: np.full(len(alphas), np.nan) for name, _ in detectors}

    print(f"{'alpha':>10}  " + "  ".join(f"{name:>14}" for name, _ in detectors))
    print("-" * (12 + 16 * len(detectors)))

    for i, pt in enumerate(sweep_data):
        alpha  = pt['alpha']
        a_star = pt.get('a_star', None)
        levels = pt['levels']
        row    = f"{alpha:>10.5g}  "
        for det_name, det in detectors:
            d = best_d_max_for_alpha(
                levels, det, rho_star, level_label,
                alpha=alpha, a_star=a_star,
            )
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
    """Convert rate density [yr^-1 kpc^-3] to 1-event/year radius [kpc]."""
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

    Returns two scalar distances (with and without a spin cut), which
    become horizontal lines on the alpha plot.
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
        print(f"  Merger reach (a* >= {a_star_threshold}):  {reach_spin:.3e} kpc")
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
    Plot d_max vs alpha (linear x, log y) with axion mass on a top axis.

    Parameters
    ----------
    alphas        : np.ndarray
    reach         : dict -- {detector_name: d_max array [kpc]}
    M_BH_solar    : float -- fixed BH mass [M_sun], used to scale top axis
    process_label : str   -- passed in but not used as a title
    show_merger_reach     : bool
    merger_*              : merger rate parameters
    xlim, ylim    : tuple or None
    savepath      : str or None

    Returns
    -------
    fig, ax
    """
    plt.rcParams.update({
        "text.usetex"        : True,
        "font.family"        : "serif",
        "font.serif"         : ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.subplots_adjust(top=0.88)

    # ── Detector reach curves (linear x, log y) ───────────────────────────────
    for det_name, d_arr in reach.items():
        style = _DET_STYLE.get(det_name, {'color': 'gray', 'ls': '-'})
        mask  = np.isfinite(d_arr) & (d_arr > 0)
        if not mask.any():
            print(f"  [INFO] {det_name}: no finite d_max values to plot.")
            continue
        ax.semilogy(alphas[mask], d_arr[mask],
                    color=style['color'],
                    ls=style['ls'],
                    linewidth=2.0,
                    label=det_name)

    # ── PBH merger reach -- horizontal line ───────────────────────────────────
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

    # ── Top x-axis: axion mass mu [eV] = alpha / r_g(M_BH) ───────────────────
    # r_g is constant for fixed M_BH, so mu is a linear rescaling of alpha.
    r_g_m      = _G_SI * M_BH_solar * _M_SUN_KG / _C_SI**2   # [m]
    r_g_eV_inv = r_g_m / _HBAR_C_EV_M                         # [eV^-1]

    ax_top = ax.secondary_xaxis(
        'top',
        functions=(
            lambda a: np.asarray(a, dtype=float) / r_g_eV_inv,
            lambda mu: np.asarray(mu, dtype=float) * r_g_eV_inv,
        ),
    )
    ax_top.set_xlabel(r'$\mu_a\ [\mathrm{eV}]$', fontsize=13, labelpad=6)

    # ── Legend and grid ───────────────────────────────────────────────────────
    ax.legend(fontsize=9, loc='best', frameon=False)
    ax.grid(True, which='major', alpha=0.25, linestyle='--')
    ax.grid(True, which='minor', alpha=0.10, linestyle=':', axis='y')

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
# Level shorthand resolver
# ═════════════════════════════════════════════════════════════════════════════

# Spectroscopic letter → azimuthal quantum number l
_SPEC_TO_L = {
    's': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4,
    'h': 5, 'i': 6, 'j': 7, 'k': 8, 'l': 9,
}


def resolve_level(shorthand):
    """
    Convert a human-readable spectroscopic shorthand into the process type
    and the full label string used in the .dat files.

    All superradiant modes have m = l, so only n and the spectroscopic
    letter are needed to fully specify a level.

    Parameters
    ----------
    shorthand : str
        One or two space-separated tokens of the form '<n><letter>'.
        One token  → annihilation.   e.g. '2p'      → |211⟩
        Two tokens → transition, excited state first.
                     e.g. '6g 5g'   → |644⟩→|544⟩

    Returns
    -------
    process : str   -- 'annihilation' or 'transitions'
    label   : str   -- full label matching the .dat file column

    Examples
    --------
    >>> resolve_level('2p')
    ('annihilation', '|211⟩')

    >>> resolve_level('3d')
    ('annihilation', '|322⟩')

    >>> resolve_level('6g 5g')
    ('transitions', '|644⟩→|544⟩')

    >>> resolve_level('4f 3d')
    ('transitions', '|433⟩→|322⟩')

    Raises
    ------
    ValueError for unrecognised tokens or more than two states.
    """
    def _parse_one(token):
        token = token.strip().lower()
        if len(token) < 2:
            raise ValueError(f"Cannot parse level token {token!r}: expected e.g. '2p'")
        n_str, letter = token[:-1], token[-1]
        if letter not in _SPEC_TO_L:
            raise ValueError(
                f"Unknown spectroscopic letter {letter!r}. "
                f"Known: {', '.join(sorted(_SPEC_TO_L))}"
            )
        try:
            n = int(n_str)
        except ValueError:
            raise ValueError(f"Cannot parse principal quantum number from {token!r}")
        l = _SPEC_TO_L[letter]
        m = l          # superradiant modes always have m = l
        return f'|{n}{l}{m}\u27e9'   # e.g. '|211⟩'

    tokens = shorthand.strip().split()
    if len(tokens) == 1:
        return 'annihilation', _parse_one(tokens[0])
    elif len(tokens) == 2:
        label = _parse_one(tokens[0]) + '\u2192' + _parse_one(tokens[1])
        return 'transitions', label
    else:
        raise ValueError(
            f"resolve_level expects 1 or 2 tokens, got {len(tokens)}: {shorthand!r}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Main -- edit ONLY this block
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Fixed physical parameters ─────────────────────────────────────────────
    M_BH_SOLAR = 1e-6           # BH mass [solar masses] -- fixed across the sweep

    # ── Level selection ───────────────────────────────────────────────────────
    # Use spectroscopic shorthand: '<n><letter>' for annihilation,
    # '<n><letter> <n><letter>' for a transition (excited state first).
    # Set to None to plot the envelope over all levels of both processes.
    #
    # Examples:
    #   LEVEL_SHORTHAND = '2p'       →  annihilation  |211⟩
    #   LEVEL_SHORTHAND = '3d'       →  annihilation  |322⟩
    #   LEVEL_SHORTHAND = '6g 5g'    →  transitions   |644⟩→|544⟩
    #   LEVEL_SHORTHAND = '4f 3d'    →  transitions   |433⟩→|322⟩
    LEVEL_SHORTHAND = '6g 5g'

    if LEVEL_SHORTHAND is not None:
        PROCESS, LEVEL = resolve_level(LEVEL_SHORTHAND)
        print(f"Resolved '{LEVEL_SHORTHAND}'  →  process='{PROCESS}'  label='{LEVEL}'")
    else:
        PROCESS = 'annihilation'   # fallback when plotting all levels
        LEVEL   = None
    # ── Data directory (output of run_alpha_sweep in superradiance_simulation.py)
    DATA_DIR = 'Sem 2/8. Numerical Simulations/Data/alpha_sweep'

    # ── Detection threshold ───────────────────────────────────────────────────
    RHO_STAR = 1.0              # SNR threshold rho*

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

    level_str = f' -- level {LEVEL}' if LEVEL is not None else ' (all levels)'
    process_label = (
        rf'Axion cloud annihilation{level_str}'
        if PROCESS == 'annihilation'
        else rf'Axion cloud transitions{level_str}'
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"Alpha reach sweep  --  {PROCESS}  --  M_BH = {M_BH_SOLAR} M_sun")
    print("=" * 65)
    print(f"\nLoading alpha-sweep data from '{DATA_DIR}'...")
    sweep_data = load_all_sweep_files(DATA_DIR, process=PROCESS,
                                      mass_filter=M_BH_SOLAR)

    # ── Compute reach for all detectors ───────────────────────────────────────
    print("Computing d_max(alpha) for all detectors...\n")
    alphas, reach = run_alpha_reach_sweep(
        sweep_data, rho_star=RHO_STAR, include_ligo=True,
        level_label=LEVEL,
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