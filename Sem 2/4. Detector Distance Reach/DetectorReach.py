"""
DetectorReach.py
================

Computes and plots the maximum detectable distance d_max(M_BH) for
superradiance gravitational-wave sources — both annihilation and transition
processes — across all detectors defined in GWDetectors.py.

Detectors included
------------------
- ADMX-EFR         (Magnetic Weber Bar, broadband)
- DMRadio-GUT      (Magnetic Weber Bar, broadband)
- LIGO HF          (Schnabel & Korobko 2024 power-law extrapolation)

Imports
-------
All detector classes, instances, and noise PSD functions come from
GWDetectors.py.  Physics (ParamCalculator, SuperradianceRateCF) is resolved
via the same directory-walking approach used in the original scripts.

Usage
-----
Edit the "USER PARAMETERS" block at the bottom and run:

    python DetectorReach.py
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Import all detector machinery from GWDetectors.py
# ─────────────────────────────────────────────────────────────────────────────

from GWDetectors import (
    # Constants
    G_N_NAT, EV_TO_SI, INV_EV_TO_M, KPC_TO_M,
    # MWB
    MagneticWeberBar, ADMX_EFR, DMRADIO_GUT, MWB_DETECTORS,
    mwb_noise_psd,
    # IFO
    IFOConfig, IFO_DETECTORS,
    ifo_noise_psd,
)

# ─────────────────────────────────────────────────────────────────────────────
# Resolve physics module paths (ParamCalculator, SuperradianceRateCF)
# ─────────────────────────────────────────────────────────────────────────────

current_dir = Path(__file__).resolve().parent
sem2_dir    = None
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
    G_N,
    calc_rg_from_bh_mass,
    calc_omega_transition,
    calc_omega_ann,
    calc_transition_rate,
    calc_annihilation_rate,
    calc_n_max,
    calc_delta_astar,
    calc_char_t_ann,
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


# ═════════════════════════════════════════════════════════════════════════════
# Source factories
# Both return (freq_func, h_func, tau_func) with the uniform interface:
#   freq_func(M_solar)     -> f [Hz]
#   h_func(f, M_solar)     -> h_unit [dimensionless strain at r = 1 m]
#   tau_func(f, M_solar)   -> tau [s]
#
# d_max = h_unit * sqrt(tau / S_h_noise)   (for both source types)
# ═════════════════════════════════════════════════════════════════════════════

def make_transition_source(alpha, transition, n_e, n_g, filepath, debug=False):
    """
    Return (freq_func, h_func, tau_func) for a superradiance transition.

    Parameters
    ----------
    alpha      : float — gravitational fine-structure constant
    transition : str   — level string, e.g. '3p 2p'
    n_e        : int   — excited-state principal quantum number
    n_g        : int   — ground-state principal quantum number
    filepath   : str   — path to superradiance rate .dat file
    debug      : bool  — print intermediate values for the first mass point

    Physics
    -------
    h_peak(r) = sqrt(4 G_N Gamma_sr^2 / (omega_t Gamma_t)) / r
    tau        = 1 / Gamma_sr
    """
    _debug_done = [False]

    def freq_func(M_solar):
        r_g = calc_rg_from_bh_mass(M_solar)
        omega_t = calc_omega_transition(r_g, alpha, n_e, n_g)   # [eV]
        return float(omega_t) * EV_TO_SI / (2 * np.pi)          # [Hz]

    def h_func(f_t, M_solar):
        r_g         = calc_rg_from_bh_mass(M_solar)             # [eV^-1]
        omega_t_eV  = 2 * np.pi * f_t / EV_TO_SI               # Hz -> eV

        # Rates in natural units [eV]
        Gamma_t_eV  = calc_transition_rate(
                          transition, alpha, omega_t_eV, G_N, r_g
                      )
        sr_out      = sr_rate_dimensioned(
                          alpha, M_solar, filepath=filepath, method='cf'
                      )
        Gamma_sr_eV = sr_out['gamma_natural_eV']                # [eV]

        # h(r) = sqrt(4 G_N Gamma_sr^2 / (omega_t Gamma_t)) / r
        # Evaluate at r = 1 eV^-1 (all natural units), then convert to r = 1 m
        h_nat  = np.sqrt(
            4 * G_N * float(Gamma_sr_eV)**2
            / (float(omega_t_eV) * float(Gamma_t_eV))
        )                                                        # strain at r=1 eV^-1
        h_unit = h_nat * INV_EV_TO_M                            # strain at r=1 m

        if debug and not _debug_done[0]:
            Gamma_sr_SI = sr_out['gamma_SI']
            print(f"\n{'─'*60}")
            print(f"[TRANS DEBUG] M_solar      = {M_solar:.4e} Msun")
            print(f"[TRANS DEBUG] r_g          = {r_g:.4e} eV^-1")
            print(f"[TRANS DEBUG] f_t          = {f_t:.4e} Hz")
            print(f"[TRANS DEBUG] omega_t      = {float(omega_t_eV):.4e} eV")
            print(f"[TRANS DEBUG] Gamma_t      = {float(Gamma_t_eV):.4e} eV")
            print(f"[TRANS DEBUG] Gamma_sr     = {float(Gamma_sr_eV):.4e} eV")
            print(f"[TRANS DEBUG] Gamma_sr(SI) = {Gamma_sr_SI:.4e} s^-1")
            print(f"[TRANS DEBUG] h_nat(r=1eV^-1) = {h_nat:.4e}")
            print(f"[TRANS DEBUG] h_unit(r=1m)    = {h_unit:.4e}")
            print(f"[TRANS DEBUG] h at 1 kpc      = {h_unit/KPC_TO_M:.4e}")
            print(f"{'─'*60}\n")
            _debug_done[0] = True

        return h_unit

    def tau_func(f_t, M_solar):
        sr_out = sr_rate_dimensioned(
                     alpha, M_solar, filepath=filepath, method='cf'
                 )
        return 1.0 / sr_out['gamma_SI']                         # [s]

    return freq_func, h_func, tau_func


def make_annihilation_source(alpha, level, n, l, m, astar_init, debug=False):
    """
    Return (freq_func, h_func, tau_func) for an axion cloud annihilation.

    Parameters
    ----------
    alpha      : float — gravitational fine-structure constant
    level      : str   — level string, e.g. '2p'
    n          : int   — principal quantum number
    l          : int   — orbital quantum number
    m          : int   — azimuthal quantum number
    astar_init : float — initial BH spin parameter
    debug      : bool  — print intermediate values for the first mass point

    Physics
    -------
    h_peak(r) = sqrt(8 G_N Gamma_ann / omega_ann) * N_max / r
    tau        = 1 / (N_max * Gamma_ann)   [as defined in ParamCalculator]
    """
    _debug_done = [False]

    def freq_func(M_solar):
        r_g     = calc_rg_from_bh_mass(M_solar)
        omega_a = calc_omega_ann(r_g, alpha, n)                  # [eV]
        return float(omega_a) * EV_TO_SI / (2 * np.pi)          # [Hz]

    def h_func(f_t, M_solar):
        r_g          = calc_rg_from_bh_mass(M_solar)            # [eV^-1]
        omega_ann    = calc_omega_ann(r_g, alpha, n)             # [eV]
        ann_rate     = calc_annihilation_rate(
                           level, alpha, omega_ann, G_N, r_g
                       )                                         # [eV]
        bh_mass_eV   = r_g / G_N_NAT
        delta_astar  = calc_delta_astar(astar_init, r_g, alpha, n, m)
        n_max        = calc_n_max(bh_mass_eV, delta_astar, m)

        # h(r) = sqrt(8 G_N Gamma_ann / omega_ann) * N_max / r
        h_nat  = float(n_max) * np.sqrt(
                     8 * G_N * float(ann_rate) / float(omega_ann)
                 )                                               # strain at r=1 eV^-1
        h_unit = h_nat * INV_EV_TO_M                            # strain at r=1 m

        if debug and not _debug_done[0]:
            print(f"\n{'─'*60}")
            print(f"[ANN DEBUG] M_solar        = {M_solar:.4e} Msun")
            print(f"[ANN DEBUG] r_g            = {r_g:.4e} eV^-1")
            print(f"[ANN DEBUG] omega_ann      = {float(omega_ann):.4e} eV")
            print(f"[ANN DEBUG] f_ann          = {f_t:.4e} Hz")
            print(f"[ANN DEBUG] ann_rate       = {float(ann_rate):.4e} eV")
            print(f"[ANN DEBUG] delta_astar    = {delta_astar:.4f}")
            print(f"[ANN DEBUG] n_max          = {float(n_max):.4e}")
            print(f"[ANN DEBUG] h_nat(r=1eV^-1)= {h_nat:.4e}")
            print(f"[ANN DEBUG] h_unit(r=1m)   = {h_unit:.4e}")
            print(f"[ANN DEBUG] h at 1 kpc     = {h_unit/KPC_TO_M:.4e}")
            print(f"{'─'*60}\n")
            _debug_done[0] = True

        return h_unit

    def tau_func(f_t, M_solar):
        r_g         = calc_rg_from_bh_mass(M_solar)
        omega_ann   = calc_omega_ann(r_g, alpha, n)
        ann_rate    = calc_annihilation_rate(level, alpha, omega_ann, G_N, r_g)
        bh_mass_eV  = r_g / G_N_NAT
        delta_astar = calc_delta_astar(astar_init, r_g, alpha, n, m)
        n_max       = calc_n_max(bh_mass_eV, delta_astar, m)
        tau_eV      = calc_char_t_ann(ann_rate, n_max)          # [eV^-1]
        return float(tau_eV) / EV_TO_SI                         # [s]

    return freq_func, h_func, tau_func


# ═════════════════════════════════════════════════════════════════════════════
# Core single-point computation (detector-agnostic)
# ═════════════════════════════════════════════════════════════════════════════

def _get_noise(det, f_t):
    """
    Return S_h_noise [Hz^-1] at frequency f_t for either a MWB or IFO detector.

    Parameters
    ----------
    det : MagneticWeberBar | str
        MagneticWeberBar instance, or IFO key string (e.g. 'adv_ligo').
    f_t : float
        Frequency [Hz].
    """
    if isinstance(det, MagneticWeberBar):
        return mwb_noise_psd(det, np.array([f_t]))[0]
    elif isinstance(det, str):
        val = ifo_noise_psd(np.array([f_t]), detector_key=det)[0]
        return val   # may be NaN below f_FSR
    else:
        raise TypeError(f"det must be MagneticWeberBar or str, got {type(det)}")


def compute_point(M_solar, freq_func, h_func, tau_func,
                  det, f_band, rho_star=1.0):
    """
    Compute (f, d_max_kpc) for a single BH mass and detector.

    Returns (f, d_max_kpc, h_unit, tau, S_h_noise).
    Any failure returns np.nan for d_max_kpc; f is returned when known.
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

    S_h_noise = _get_noise(det, f_t)
    if not np.isfinite(S_h_noise) or S_h_noise <= 0:
        return f_t, np.nan, h_unit, tau_val, np.nan

    d_max_m   = h_unit * np.sqrt(tau_val / S_h_noise) / rho_star
    d_max_kpc = d_max_m / KPC_TO_M

    if not np.isfinite(d_max_kpc) or d_max_kpc <= 0:
        return f_t, np.nan, h_unit, tau_val, S_h_noise

    return f_t, d_max_kpc, h_unit, tau_val, S_h_noise


# ═════════════════════════════════════════════════════════════════════════════
# Mass sweep — runs coarse pass then dense pass near resonance for each detector
# ═════════════════════════════════════════════════════════════════════════════

def _single_det_sweep(freq_func, h_func, tau_func,
                      det, f_band, M_range,
                      n_coarse, n_dense, rho_star):
    """
    Two-stage mass sweep for one detector.

    Returns dict with keys: f, d, M (coarse, valid only),
    d_combined, M_combined (coarse + dense), f_res, d_res, M_res.
    """
    def _sweep(M_array):
        fs, ds, Ms = [], [], []
        for M in M_array:
            f, d, _, _, _ = compute_point(
                M, freq_func, h_func, tau_func, det, f_band, rho_star
            )
            fs.append(f); ds.append(d); Ms.append(M)
        return np.array(fs), np.array(ds), np.array(Ms)

    M_coarse            = np.logspace(np.log10(M_range[0]),
                                      np.log10(M_range[1]), n_coarse)
    f_c, d_c, M_c       = _sweep(M_coarse)
    valid_c             = np.isfinite(d_c)

    M_res, f_res, d_res = np.nan, np.nan, np.nan
    M_fine, d_fine      = None, None

    if valid_c.any():
        # For MWB detectors find the point closest to f_mech; for IFOs find the peak.
        if isinstance(det, MagneticWeberBar):
            idx = np.argmin(np.abs(f_c[valid_c] - det.f_mech))
        else:
            idx = np.argmax(d_c[valid_c])

        M_guess = M_c[valid_c][idx]
        M_dense = np.logspace(np.log10(M_guess / 10),
                              np.log10(M_guess * 10), n_dense)
        f_d, d_d, _ = _sweep(M_dense)
        valid_d = np.isfinite(d_d)
        if valid_d.any():
            M_fine = M_dense[valid_d]
            d_fine = d_d[valid_d]
            idx_res = np.argmax(d_fine)
            M_res   = M_fine[idx_res]
            f_res   = f_d[valid_d][idx_res]
            d_res   = d_fine[idx_res]

    # Combine coarse + dense
    if M_fine is not None:
        M_comb = np.concatenate([M_c[valid_c], M_fine])
        d_comb = np.concatenate([d_c[valid_c], d_fine])
        order  = np.argsort(M_comb)
        M_comb, d_comb = M_comb[order], d_comb[order]
        _, uniq = np.unique(M_comb, return_index=True)
        M_comb, d_comb = M_comb[uniq], d_comb[uniq]
    else:
        M_comb = M_c[valid_c]
        d_comb = d_c[valid_c]

    return {
        'f': f_c[valid_c],
        'd': d_c[valid_c],
        'M': M_c[valid_c],
        'M_combined': M_comb,
        'd_combined': d_comb,
        'f_res': f_res,
        'd_res': d_res,
        'M_res': M_res,
    }


def run_sweep(freq_func, h_func, tau_func,
              M_range  = (1e-12, 1e4),
              n_coarse = 300,
              n_dense  = 800,
              f_band   = (1e2, 1e8),
              rho_star = 1.0,
              include_ligo: bool = True):
    """
    Run the two-stage mass sweep for every detector.

    Returns
    -------
    results : dict
        Keys are detector labels.
        Each value is the dict returned by _single_det_sweep, plus 'name'.

        MWB detectors : 'ADMX-EFR', 'DMRadio-GUT'
        IFO detectors : 'LIGO HF'   (if include_ligo=True)
    """
    results = {}

    # MWB detectors
    mwb_f_band = f_band
    for det in MWB_DETECTORS:
        print(f"  Sweeping {det.name}...")
        data = _single_det_sweep(
            freq_func, h_func, tau_func,
            det, mwb_f_band, M_range, n_coarse, n_dense, rho_star,
        )
        data['name'] = det.name
        results[det.name] = data

    # IFO detectors
    if include_ligo:
        for key, ifo in IFO_DETECTORS.items():
            print(f"  Sweeping {ifo.name}...")
            if key == 'ligo_data':
                ifo_f_band = (10, 1000)  # adjust to your file range
            else:
                ifo_f_band = (ifo.f_FSR, 1e11)
            data = _single_det_sweep(
                freq_func, h_func, tau_func,
                key, ifo_f_band, M_range, n_coarse, n_dense, rho_star,
            )
            data['name'] = ifo.name
            results[ifo.name] = data

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

# Colour and style assigned per detector name
_DET_STYLE = {
    'ADMX-EFR'   : {'color': 'steelblue', 'ls': '-'},
    'DMRadio-GUT': {'color': 'teal',       'ls': '-'},
    'LIGO HF'    : {'color': 'black',     'ls': '--'},
    'LIGO'       : {'color': 'black',      'ls': '-'},
}

def plot_reach(results, alpha, process_label,
               savepath=None,
               show_noise_curves=True,
               xlim=None, ylim=None):
    """
    Plot d_max vs M_BH for all detectors in results.

    Parameters
    ----------
    results : dict
        Output of run_sweep().
    alpha : float
        Gravitational coupling (for title).
    process_label : str
        LaTeX string describing the process (for title/legend).
    savepath : str, optional
        Save path for the PDF.
    show_noise_curves : bool
        If True, add a right-hand axis with sqrt(S_h^noise) vs M_BH.
    xlim, ylim : tuple, optional
        Axis limits.  Defaults chosen automatically.
    """
    plt.rcParams.update({
        "text.usetex"        : True,
        "font.family"        : "serif",
        "font.serif"         : ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    fig, ax1 = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(top=0.82)

    # ── Distance reach curves ─────────────────────────────────────────────────
    for name, data in results.items():
        M_plot = data['M_combined']
        d_plot = data['d_combined']
        if len(M_plot) == 0:
            continue

        style = _DET_STYLE.get(name, {'color': 'gray', 'ls': '-'})
        ax1.loglog(M_plot, d_plot,
                   color=style['color'], linewidth=2.0,
                   linestyle=style['ls'], label=name)

        # Vertical line at resonance mass
        if np.isfinite(data['d_res']) and np.isfinite(data['M_res']):
            ax1.axvline(data['M_res'], color=style['color'],
                        linewidth=0.8, linestyle=':', alpha=0.5)

    # ── Noise curves on twin axis ─────────────────────────────────────────────
    if show_noise_curves:
        # Build a frequency-to-mass map from ADMX-EFR coarse data
        admx_data = results.get('ADMX-EFR', {})
        f_c = admx_data.get('f', np.array([]))
        M_c = admx_data.get('M', np.array([]))
        valid = np.isfinite(f_c) & np.isfinite(M_c)

        if valid.sum() >= 2:
            from scipy.interpolate import interp1d

            order = np.argsort(f_c[valid])
            f_s   = f_c[valid][order]
            M_s   = M_c[valid][order]
            _, ui = np.unique(f_s, return_index=True)
            f_u, M_u = f_s[ui], M_s[ui]

            if len(f_u) >= 2:
                f2M = interp1d(np.log10(f_u), np.log10(M_u),
                               bounds_error=False, fill_value=np.nan)

                ax_noise = ax1.twinx()
                freqs_n  = np.logspace(2, 10, 5000)
                log_M_n  = f2M(np.log10(freqs_n))
                ok       = np.isfinite(log_M_n)

                if ok.any():
                    M_n = 10.0**log_M_n[ok]
                    f_n = freqs_n[ok]
                    order2 = np.argsort(M_n)
                    M_n, f_n = M_n[order2], f_n[order2]

                    noise_specs = [
                        ('ADMX-EFR',
                         lambda f: mwb_noise_psd(ADMX_EFR,    np.asarray(f)),
                         'gray', '--'),
                        ('DMRadio-GUT',
                         lambda f: mwb_noise_psd(DMRADIO_GUT, np.asarray(f)),
                         'darkgray', ':'),
                        ('LIGO HF',
                         lambda f: ifo_noise_psd(np.asarray(f), 'adv_ligo'),
                         'plum', '-.'),
                    ]
                    for det_label, noise_fn, col, ls in noise_specs:
                        Sn = np.sqrt(noise_fn(f_n))
                        fin = np.isfinite(Sn)
                        if fin.any():
                            ax_noise.loglog(
                                M_n[fin], Sn[fin],
                                color=col, linewidth=1.0, linestyle=ls,
                                alpha=0.6,
                                label=fr'{det_label} $\sqrt{{S_h^\mathrm{{noise}}}}$',
                            )

                ax_noise.set_ylabel(
                    r'$\left(S_h^\mathrm{noise}\right)^{1/2}\ [\mathrm{Hz}^{-1/2}]$',
                    fontsize=10, color='gray',
                )
                ax_noise.tick_params(axis='y', labelcolor='gray')

                # Merge legends
                lines1, labs1 = ax1.get_legend_handles_labels()
                lines2, labs2 = ax_noise.get_legend_handles_labels()
                seen = {}
                for ln, lb in zip(lines1 + lines2, labs1 + labs2):
                    if lb not in seen:
                        seen[lb] = ln
                ax1.legend(seen.values(), seen.keys(),
                           fontsize=7, loc='upper left', frameon=False)
        else:
            ax1.legend(fontsize=9, loc='upper left', frameon=False)
    else:
        ax1.legend(fontsize=9, loc='upper left', frameon=False)

    # ── Axis labels, limits, title ────────────────────────────────────────────
    ax1.set_xlabel(r'$M_\mathrm{BH}\ [M_\odot]$', fontsize=13)
    ax1.set_ylabel(r'$d_\mathrm{max}\ [\mathrm{kpc}]$', fontsize=13)

    if xlim is not None:
        ax1.set_xlim(*xlim)
    if ylim is not None:
        ax1.set_ylim(*ylim)

    """title = (
        process_label + '\n'
        + fr'$\alpha = {alpha}$'
    )
    ax1.set_title(title, fontsize=10, pad=6)"""
    ax1.grid(True, which='both', alpha=0.25)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f'[saved] {savepath}')

    plt.show()
    return fig, ax1


# ═════════════════════════════════════════════════════════════════════════════
# Main — edit ONLY this block
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Common parameters ─────────────────────────────────────────────────────
    alpha      = 0.01
    astar_init = 0.99
    M_range    = (1e-12, 1e4)
    rho_star   = 1.0               # SNR threshold
    savedir    = '4. Detector Distance Reach/Plots'

    # ── Choose process: 'transition' or 'annihilation' ────────────────────────
    PROCESS = 'transition'         # <─── change this

    # ─────────────────────────────────────────────────────────────────────────
    if PROCESS == 'transition':
        # Transition: |n_e l_e m_e> -> |n_g l_g m_g>
        transition    = '3p 2p'
        n_e, n_g      = 3, 2       # principal quantum numbers
        process_label = r'$|322\rangle \to |211\rangle$ (transition)'
        filepath      = ("2. Relativistic Superradiance Rate/Mathematica/"
                         "SR_n2l1m1_at0.990_aMin0.010_aMax0.500_20260310.dat")
        f_band        = (1e2, 1e8)
        savepath      = f'{savedir}/reach_transition.pdf'

        print("Building transition source functions...")
        freq_func, h_func, tau_func = make_transition_source(
            alpha, transition, n_e, n_g, filepath, debug=False,
        )

    elif PROCESS == 'annihilation':
        # Annihilation: |n l m>
        level         = '2p'
        n, l, m       = 2, 1, 1
        process_label = r'$|211\rangle$ (annihilation)'
        f_band        = (1e2, 1e8)
        savepath      = f'{savedir}/reach_annihilation.pdf'

        print("Building annihilation source functions...")
        freq_func, h_func, tau_func = make_annihilation_source(
            alpha, level, n, l, m, astar_init, debug=False,
        )

    else:
        raise ValueError(f"Unknown PROCESS: {PROCESS!r}")

    # ── Run sweep across all detectors ────────────────────────────────────────
    print(f"\nRunning mass sweep ({PROCESS})...")
    results = run_sweep(
        freq_func, h_func, tau_func,
        M_range      = M_range,
        f_band       = f_band,
        rho_star     = rho_star,
        include_ligo = True,
    )

    # ── Print peak distances ──────────────────────────────────────────────────
    print("\nPeak distance reaches:")
    print("-" * 45)
    for name, data in results.items():
        if np.isfinite(data['d_res']):
            print(f"  {name:<15}: {data['d_res']:.3e} kpc  "
                  f"at M = {data['M_res']:.3e} Msun  "
                  f"(f = {data['f_res']:.3e} Hz)")
        else:
            print(f"  {name:<15}: no detectable signal in sweep range")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\nPlotting...")
    plot_reach(
        results, alpha, process_label,
        savepath          = savepath,
        show_noise_curves = False,
    )
