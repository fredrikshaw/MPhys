import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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

# ─────────────────────────────────────────────────────────────────────────────
# Directory resolution and imports
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
# Source function factories
#
# Each factory takes the physical parameters for a given process and returns
# three callables with the standard interface expected by compute_point:
#
#   freq_func(M_solar)       -> f  [Hz]
#   h_func(f, M_solar)       -> h  [m]       strain at unit distance r = 1 m
#   tau_func(f, M_solar)     -> tau [s]       characteristic signal duration
#
# The factory pattern means compute_point never needs to know whether it is
# computing a transition or an annihilation — it just calls these three
# functions with the same interface.
# ─────────────────────────────────────────────────────────────────────────────

def make_transition_funcs(alpha, transition, filepath):
    """
    Return (freq_func, h_func, tau_func) for a superradiance transition.

    The GW frequency is set by the energy difference between the two levels:
        f_t = omega_t / 2pi

    Parameters
    ----------
    alpha      : float — dimensionless gravitational coupling
    transition : str   — transition string e.g. '4d 3d'
    filepath   : str   — path to SR data file

    Returns
    -------
    freq_func : M_solar -> f_t [Hz]
    h_func    : (f_t, M_solar) -> h_unit [m]
    tau_func  : (f_t, M_solar) -> tau [s]
    """

    def freq_func(M_solar):
        r_g_nat     = calc_rg_from_bh_mass(M_solar)
        omega_t_nat = calc_omega_transition(r_g_nat, alpha, 4, 3)
        return omega_t_nat * EV_TO_SI / (2 * np.pi)            # [Hz]

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
        )                                                        # [m]

    def tau_func(f_t, M_solar):
        # For transitions tau = 1 / Gamma_sr
        # f_t is accepted for interface consistency but not used
        return 1.0 / sr_rate_dimensioned(
                         alpha, M_solar, filepath=filepath, method='cf'
                     )['gamma_SI']                               # [s]

    return freq_func, h_func, tau_func


def make_annihilation_funcs(alpha, level, n, l, m, astar_init):
    """
    Return (freq_func, h_func, tau_func) for a superradiance annihilation.

    The GW frequency for annihilation is TWICE the axion rest mass frequency:
        f_ann = omega_ann / 2pi = 2 * m_a * c^2 / h
              = 2 * alpha * c^3 / (2pi * G_N * M_BH)

    This is different from the transition frequency which depends on the
    energy DIFFERENCE between two levels. Here both photons carry the full
    axion rest mass, so the frequency scales as 1/M_BH with a different
    prefactor than transitions.

    Parameters
    ----------
    alpha      : float — dimensionless gravitational coupling
    level      : str   — level string for annihilation rate calculation
    n          : int   — principal quantum number of the superradiant level
    l          : int   — orbital quantum number
    m          : int   — azimuthal quantum number
    astar_init : float — initial dimensionless BH spin parameter

    Returns
    -------
    freq_func : M_solar -> f_ann [Hz]
    h_func    : (f_ann, M_solar) -> h_unit [m]
    tau_func  : (f_ann, M_solar) -> tau [s]
    """

    def freq_func(M_solar):
        # omega_ann is computed from r_g and alpha
        # calc_omega_ann returns omega_ann in eV (natural units)
        r_g     = calc_rg_from_bh_mass(M_solar)                 # [eV^-1]
        omega   = calc_omega_ann(r_g, alpha, n)                  # [eV]
        return omega * EV_TO_SI / (2 * np.pi)                   # [Hz]

    def h_func(f_ann, M_solar):
        r_g          = calc_rg_from_bh_mass(M_solar)            # [eV^-1]
        omega_ann    = calc_omega_ann(r_g, alpha, n)             # [eV]
        ann_rate     = calc_annihilation_rate(
                           level, alpha, omega_ann,
                           G_N=6.708e-57, r_g=r_g
                       )                                         # [eV]
        bh_mass      = calc_bh_mass(r_g)                        # [eV]
        delta_a_star = calc_delta_astar(astar_init, r_g, alpha, n, m)
        n_max        = calc_n_max(bh_mass, delta_a_star, m)

        # calc_h_peak_ann returns strain in eV at unit distance r = 1 eV^-1
        # Convert: h [eV] * r [eV^-1] is dimensionless, so h [eV] at r=1 eV^-1
        # means we need to convert the reference distance to metres:
        #   1 eV^-1 = INV_EV_TO_M metres
        # h [m] = h_eV [eV] * INV_EV_TO_M
        h_eV = calc_h_peak_ann(ann_rate, omega_ann, 1, n_max)   # [eV * eV^-1] = dimensionless at r=1 eV^-1
        return float(h_eV) * INV_EV_TO_M                        # [m]

    def tau_func(f_ann, M_solar):
        r_g          = calc_rg_from_bh_mass(M_solar)            # [eV^-1]
        omega_ann    = calc_omega_ann(r_g, alpha, n)             # [eV]
        ann_rate     = calc_annihilation_rate(
                           level, alpha, omega_ann,
                           G_N=6.708e-57, r_g=r_g
                       )                                         # [eV]
        bh_mass      = calc_bh_mass(r_g)                        # [eV]
        delta_a_star = calc_delta_astar(astar_init, r_g, alpha, n, m)
        n_max        = calc_n_max(bh_mass, delta_a_star, m)

        # calc_char_t_ann returns characteristic time in eV^-1 (natural units)
        # Convert to SI: tau [s] = tau [eV^-1] / EV_TO_SI
        tau_eV = calc_char_t_ann(ann_rate, n_max)               # [eV^-1]
        return float(tau_eV) / EV_TO_SI                         # [s]

    return freq_func, h_func, tau_func


# ─────────────────────────────────────────────────────────────────────────────
# Core computation — process-agnostic
# ─────────────────────────────────────────────────────────────────────────────

def compute_point(M_solar, freq_func, h_func, tau_func,
                  det=ADMX_EFR, f_band=(1e2, 1e8)):
    """
    Compute (f, d_max_kpc, h_unit, tau_val, S_h_noise) for one BH mass.

    Works for any process — transition or annihilation — as long as the
    three callables follow the standard interface.

    Parameters
    ----------
    M_solar   : float    — BH mass [solar masses]
    freq_func : callable — freq_func(M_solar) -> f [Hz]
    h_func    : callable — h_func(f, M_solar) -> h_unit [m]
    tau_func  : callable — tau_func(f, M_solar) -> tau [s]
    det       : MagneticWeberBar
    f_band    : tuple    — (f_min, f_max) [Hz]

    Returns
    -------
    tuple : (f [Hz], d_max [kpc], h_unit [m], tau [s], S_h_noise [Hz^-1])
            Returns (f, nan, nan, nan, nan) on any failure.
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

    S_h_noise = noise_equivalent_strain_broadband(
                    det, np.array([f_t])
                )[0]

    d_max_m   = h_unit * np.sqrt(tau_val / S_h_noise)
    d_max_kpc = d_max_m / KPC_TO_M

    if not np.isfinite(d_max_kpc) or d_max_kpc <= 0:
        return f_t, np.nan, h_unit, tau_val, S_h_noise

    return f_t, d_max_kpc, h_unit, tau_val, S_h_noise


# ─────────────────────────────────────────────────────────────────────────────
# Sweep — process-agnostic
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(freq_func, h_func, tau_func,
              M_range=(1e-12, 1e1), n_coarse=300, n_dense=800,
              det=ADMX_EFR, f_band=(1e2, 1e8)):
    """
    Two-stage mass sweep, agnostic to the underlying GW process.

    Parameters
    ----------
    freq_func : callable — from make_transition_funcs or make_annihilation_funcs
    h_func    : callable
    tau_func  : callable
    M_range   : tuple   — (M_min, M_max) solar masses for coarse sweep
    n_coarse  : int
    n_dense   : int
    det       : MagneticWeberBar
    f_band    : tuple   — (f_min, f_max) [Hz]

    Returns
    -------
    dict with keys: f, d, M, fin, f_res, d_res, M_res
    """

    def _sweep(M_array):
        f_arr, d_arr, M_arr = [], [], []
        for M in M_array:
            f, d, _, _, _ = compute_point(
                M, freq_func, h_func, tau_func, det, f_band
            )
            f_arr.append(f);  d_arr.append(d);  M_arr.append(M)
            print(f"M = {M:.4e} Msun | f = {f:.4e} Hz")
        return np.array(f_arr), np.array(d_arr), np.array(M_arr)

    # Stage 1: coarse sweep
    M_coarse      = np.logspace(np.log10(M_range[0]),
                                np.log10(M_range[1]), n_coarse)
    f_c, d_c, M_c = _sweep(M_coarse)
    fin_c         = np.isfinite(d_c)

    if not fin_c.any():
        raise RuntimeError("No finite points in coarse sweep — check parameters.")

    # Stage 2: dense sweep around f_mech
    f_mech   = det.f_mech
    f_lo     = f_mech / 10.0
    f_hi     = f_mech * 10.0
    mask_res = fin_c & (f_c >= f_lo) & (f_c <= f_hi)

    if mask_res.any():
        M_lo = M_c[mask_res].min()
        M_hi = M_c[mask_res].max()
    else:
        idx_near = np.argmin(np.abs(f_c[fin_c] - f_mech))
        M_near   = M_c[fin_c][idx_near]
        M_lo     = 10**(np.log10(M_near) - 1.5)
        M_hi     = 10**(np.log10(M_near) + 1.5)

    M_dense       = np.logspace(np.log10(M_lo) - 0.3,
                                np.log10(M_hi) + 0.3, n_dense)
    f_d, d_d, M_d = _sweep(M_dense)

    # Resonance point by interpolation
    f_res_pt = np.nan;  d_res_pt = np.nan;  M_res_pt = np.nan

    sort_idx = np.argsort(f_c[fin_c])
    f_s      = f_c[fin_c][sort_idx]
    M_s      = M_c[fin_c][sort_idx]
    _, uniq  = np.unique(f_s, return_index=True)
    f_u      = f_s[uniq];  M_u = M_s[uniq]

    if len(f_u) >= 2:
        f_to_logM = interp1d(
            np.log10(f_u), np.log10(M_u),
            kind='linear', bounds_error=False, fill_value=np.nan
        )
        log_M_res = f_to_logM(np.log10(f_mech))
        if np.isfinite(log_M_res):
            M_res_exact = 10.0**log_M_res
            print(f"\nResonant mass: {M_res_exact:.4e} Msun | f = f_mech = {f_mech:.4e} Hz")
            f_r, d_r, _, _, _ = compute_point(
                M_res_exact, freq_func, h_func, tau_func, det, f_band
            )
            if np.isfinite(d_r):
                f_res_pt = f_r;  d_res_pt = d_r;  M_res_pt = M_res_exact

    # Merge and sort
    f_all = np.concatenate([f_c, f_d])
    d_all = np.concatenate([d_c, d_d])
    M_all = np.concatenate([M_c, M_d])
    sort_idx = np.argsort(f_all)
    f_all    = f_all[sort_idx]
    d_all    = d_all[sort_idx]
    M_all    = M_all[sort_idx]
    fin_all  = np.isfinite(d_all) & np.isfinite(f_all) & (f_all > 0)

    return dict(
        f=f_all, d=d_all, M=M_all, fin=fin_all,
        f_res=f_res_pt, d_res=d_res_pt, M_res=M_res_pt
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — unchanged, works for both processes
# ─────────────────────────────────────────────────────────────────────────────

def plot_reach(results, alpha, process_label,
               det=ADMX_EFR, savepath=None):
    """
    Plot the detector distance reach from the output of run_sweep().

    Parameters
    ----------
    results       : dict  — output of run_sweep()
    alpha         : float — coupling, for legend label
    process_label : str   — LaTeX string describing the process,
                            e.g. r'|422\rangle \to |322\rangle\ \text{(transition)}'
                            or   r'|211\rangle\ \text{(annihilation)}'
    det           : MagneticWeberBar
    savepath      : str or None
    """
    plt.rcParams.update({
        "text.usetex"         : True,
        "font.family"         : "serif",
        "font.serif"          : ["Computer Modern Roman"],
        "text.latex.preamble" : r"\usepackage{amsmath}"
    })

    f_all   = results['f'];    d_all   = results['d']
    M_all   = results['M'];    fin_all = results['fin']
    f_res   = results['f_res'];d_res   = results['d_res']
    M_res   = results['M_res']
    f_mech  = det.f_mech

    fig, ax1 = plt.subplots(figsize=(6, 5))
    fig.subplots_adjust(top=0.80)

    ax1.loglog(f_all[fin_all], d_all[fin_all],
               color='steelblue', linewidth=2.0,
               label=fr'${process_label},\ \alpha={alpha}$')

    if np.isfinite(d_res):
        ax1.scatter([f_res], [d_res], color='firebrick', s=60, zorder=6)
        log_M_val = np.log10(M_res)
        if abs(log_M_val - round(log_M_val)) < 0.15:
            M_label = fr'$M = 10^{{{int(round(log_M_val))}}}\,M_\odot$'
        else:
            M_label = fr'$M = {M_res:.2e}\,M_\odot$'
        ax1.annotate(
            M_label + '\n' + fr'$f = f_{{\rm mech}}$',
            xy         = (f_res, d_res),
            xytext     = (f_res * 6.0, d_res * 0.15),
            fontsize   = 9, color='firebrick',
            arrowprops = dict(arrowstyle='->', color='firebrick', lw=0.8),
        )

    ax1.axvline(f_mech, color='firebrick', linewidth=1.0,
                linestyle=':', alpha=0.7)

    ax_noise    = ax1.twinx()
    freqs_noise = np.logspace(2, 8, 2000)
    S_h         = noise_equivalent_strain_broadband(det, freqs_noise)
    ax_noise.loglog(freqs_noise, np.sqrt(S_h), color='gray',
                    linewidth=1.2, linestyle='--', alpha=0.5,
                    label=r'$\sqrt{S_h^{\rm noise}}$')
    ax_noise.set_ylabel(
        r'$\left(S_h^{\rm noise}\right)^{1/2}\ [\mathrm{Hz}^{-1/2}]$',
        fontsize=11, color='gray'
    )
    ax_noise.tick_params(axis='y', labelcolor='gray')

    f_fin = f_all[fin_all];  M_fin = M_all[fin_all]
    sort2 = np.argsort(f_fin)
    f_fin = f_fin[sort2];    M_fin = M_fin[sort2]
    _, uniq2 = np.unique(f_fin, return_index=True)
    f_fin    = f_fin[uniq2]; M_fin = M_fin[uniq2]

    log_M_min   = np.ceil(np.log10(M_fin.min()))
    log_M_max   = np.floor(np.log10(M_fin.max()))
    log_M_ticks = np.arange(log_M_min, log_M_max + 1, 1)
    f_tick_pos  = [];  lM_tick_valid = []
    for lM in log_M_ticks:
        idx_t = np.argmin(np.abs(M_fin - 10.0**lM))
        f_t_  = f_fin[idx_t]
        xl    = ax1.get_xlim()
        if xl[0] <= f_t_ <= xl[1]:
            f_tick_pos.append(f_t_)
            lM_tick_valid.append(int(lM))

    ax_top = ax1.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax1.get_xlim())
    ax_top.set_xticks(f_tick_pos)
    ax_top.set_xticklabels(
        [fr'$10^{{{lM}}}$' for lM in lM_tick_valid], fontsize=9
    )
    ax_top.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=12, labelpad=8)

    ax1.set_xlabel(r'$f\ [\mathrm{Hz}]$', fontsize=13)
    ax1.set_ylabel(r'$d_{\rm max}\ [\mathrm{kpc}]$', fontsize=13)
    ax1.grid(True, which='both', alpha=0.3)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax_noise.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2,
               fontsize=10, loc='upper right', frameon=False)

    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')

    plt.show()
    return fig, ax1


# ─────────────────────────────────────────────────────────────────────────────
# Main — edit only this block
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Common parameters ─────────────────────────────────────────────────────
    alpha   = 0.75
    M_range = (1e-12, 1e1)   # solar masses
    savepath= '4. Detector Distance Reach/distance_reach_mass_sweep.pdf'

    # =========================================================================
    # OPTION A: Transition process
    # =========================================================================
    filepath         = "2. Relativistic Superradiance Rate/Mathematica/SR_n4l2m2_at0.990_aMin0.010_aMax1.200_20260317.dat"
    transition       = '4d 3d'
    process_label_t  = r'|422\rangle \to |322\rangle\ \text{(transition)}'

    freq_func_t, h_func_t, tau_func_t = make_transition_funcs(
        alpha      = alpha,
        transition = transition,
        filepath   = filepath,
    )

    results_t = run_sweep(
        freq_func = freq_func_t,
        h_func    = h_func_t,
        tau_func  = tau_func_t,
        M_range   = M_range,
    )

    plot_reach(
        results       = results_t,
        alpha         = alpha,
        process_label = process_label_t,
        savepath      = savepath,
    )

    # =========================================================================
    # OPTION B: Annihilation process
    # =========================================================================
    level          = '2p'    # level string for annihilation rate
    n, l, m        = 2, 1, 1  # quantum numbers of superradiant level
    astar_init     = 0.99     # initial BH spin
    process_label_a= r'|211\rangle\ \text{(annihilation)}'

    freq_func_a, h_func_a, tau_func_a = make_annihilation_funcs(
        alpha      = alpha,
        level      = level,
        n          = n,
        l          = l,
        m          = m,
        astar_init = astar_init,
    )

    results_a = run_sweep(
        freq_func = freq_func_a,
        h_func    = h_func_a,
        tau_func  = tau_func_a,
        M_range   = M_range,
    )

    plot_reach(
        results       = results_a,
        alpha         = alpha,
        process_label = process_label_a,
        savepath      = savepath.replace('.pdf', '_ann.pdf'),
    )