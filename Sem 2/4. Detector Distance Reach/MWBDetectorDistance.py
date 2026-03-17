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

# ─────────────────────────────────────────────────────────────────────────────
# Source functions
# ─────────────────────────────────────────────────────────────────────────────

def h_peak(f_t_SI, transition, alpha, M_solar, filepath):
    """
    Peak strain at unit distance r = 1 m.

    Parameters
    ----------
    f_t_SI     : float  — transition frequency [Hz]
    transition : str    — transition string e.g. '4d 3d'
    alpha      : float  — dimensionless gravitational coupling
    M_solar    : float  — BH mass [solar masses]
    filepath   : str    — path to SR data file

    Returns
    -------
    float — strain at r = 1 m  [m * dimensionless]
    """
    r_g_nat     = calc_rg_from_bh_mass(M_solar)
    r_g_SI      = r_g_nat * INV_EV_TO_M
    Gamma_t_nat = calc_transition_rate(transition, alpha, f_t_SI, G_NEWTON, r_g_SI)
    Gamma_t_SI  = Gamma_t_nat * EV_TO_SI
    Gamma_sr_SI = sr_rate_dimensioned(
                      alpha, M_solar, filepath=filepath, method='cf'
                  )['gamma_SI']
    omega_t_SI  = 2 * np.pi * f_t_SI
    return np.sqrt(4 * G_NEWTON / omega_t_SI * Gamma_sr_SI**2 / Gamma_t_SI)


def tau(f_t_SI, alpha, M_solar, filepath):
    """
    Signal duration tau_t = 1 / Gamma_sr  [s]

    Parameters
    ----------
    f_t_SI   : float  — transition frequency [Hz] (unused, kept for interface consistency)
    alpha    : float  — dimensionless gravitational coupling
    M_solar  : float  — BH mass [solar masses]
    filepath : str    — path to SR data file

    Returns
    -------
    float — signal duration [s]
    """
    return 1.0 / sr_rate_dimensioned(
               alpha, M_solar, filepath=filepath, method='cf'
           )['gamma_SI']



def h_peak_ann(f_t_SI, level, alpha, M_solar, n, l, m, astar_init):

    omega_ann = calc_omega_ann(r_g, alpha, n)
    r_g = calc_rg_from_bh_mass(M_solar)

    ann_rate = calc_annihilation_rate(level, alpha, omega_ann, G_N=6.708e-57, r_g=r_g)

    bh_mass = calc_bh_mass(r_g) #[eV]
    delta_a_star = calc_delta_astar(astar_init, r_g, alpha, n, m)

    n_max = calc_n_max(bh_mass, delta_a_star, m)

    return calc_h_peak_ann(ann_rate, omega_ann, 1, n_max) #[eV]


def tau_ann(M_solar, level, n, l, m, astar_init, alpha):
    omega_ann = calc_omega_ann(r_g, alpha, n)
    r_g = calc_rg_from_bh_mass(M_solar)

    ann_rate = calc_annihilation_rate(level, alpha, omega_ann, G_N=6.708e-57, r_g=r_g)
    bh_mass = calc_bh_mass(r_g) #[eV]
    delta_a_star = calc_delta_astar(astar_init, r_g, alpha, n, m)

    n_max = calc_n_max(bh_mass, delta_a_star, m)

    return calc_char_t_ann(ann_rate, n_max) #[eV]


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_point(M_solar, alpha, transition, filepath,
                  h_peak_func, tau_func,
                  det=ADMX_EFR, f_band=(1e2, 1e8)):
    """
    Compute (f_t, d_max_kpc, h_unit, tau_val, S_h_noise) for one BH mass.

    Parameters
    ----------
    M_solar     : float    — BH mass [solar masses]
    alpha       : float    — dimensionless gravitational coupling
    transition  : str      — transition string
    filepath    : str      — path to SR data file
    h_peak_func : callable — h_peak(f_t, transition, alpha, M_solar, filepath)
    tau_func    : callable — tau(f_t, alpha, M_solar, filepath)
    det         : MagneticWeberBar
    f_band      : tuple    — (f_min, f_max) detector band in Hz

    Returns
    -------
    tuple : (f_t [Hz], d_max [kpc], h_unit [m], tau_val [s], S_h_noise [Hz^-1])
            Any failure returns (f_t, nan, nan, nan, nan)
    """
    r_g_nat     = calc_rg_from_bh_mass(M_solar)
    omega_t_nat = calc_omega_transition(r_g_nat, alpha, 4, 3)
    omega_t_SI  = omega_t_nat * EV_TO_SI
    f_t         = omega_t_SI / (2 * np.pi)

    if not (f_band[0] <= f_t <= f_band[1]):
        return f_t, np.nan, np.nan, np.nan, np.nan

    try:
        h_unit = h_peak_func(f_t, transition, alpha, M_solar, filepath)
    except Exception:
        return f_t, np.nan, np.nan, np.nan, np.nan

    try:
        tau_val = tau_func(f_t, alpha, M_solar, filepath)
    except Exception:
        return f_t, np.nan, np.nan, np.nan, np.nan

    S_h_noise = noise_equivalent_strain_broadband(
                    det, np.array([f_t])
                )[0]

    d_max_m   = h_unit * np.sqrt(tau_val / S_h_noise)
    d_max_kpc = d_max_m / 3.086e19

    if not np.isfinite(d_max_kpc) or d_max_kpc <= 0:
        return f_t, np.nan, h_unit, tau_val, S_h_noise

    return f_t, d_max_kpc, h_unit, tau_val, S_h_noise


def run_sweep(alpha, transition, filepath, h_peak_func, tau_func,
              M_range=(1e-12, 1e1), n_coarse=300, n_dense=800,
              det=ADMX_EFR, f_band=(1e2, 1e8)):
    """
    Run the two-stage mass sweep and return merged sorted arrays.

    Parameters
    ----------
    alpha       : float   — dimensionless gravitational coupling
    transition  : str     — transition string
    filepath    : str     — path to SR data file
    h_peak_func : callable
    tau_func    : callable
    M_range     : tuple   — (M_min, M_max) in solar masses for coarse sweep
    n_coarse    : int     — number of coarse sweep points
    n_dense     : int     — number of dense sweep points around resonance
    det         : MagneticWeberBar
    f_band      : tuple   — detector frequency band [Hz]

    Returns
    -------
    dict with keys:
        'f'       : np.ndarray — transition frequencies [Hz]
        'd'       : np.ndarray — d_max [kpc]
        'M'       : np.ndarray — BH masses [solar masses]
        'fin'     : np.ndarray — boolean mask of finite points
        'f_res'   : float      — resonance point frequency [Hz]  or nan
        'd_res'   : float      — resonance point d_max [kpc]     or nan
        'M_res'   : float      — resonance point BH mass [Msun]  or nan
    """

    def _sweep(M_array):
        f_arr, d_arr, M_arr = [], [], []
        for M in M_array:
            f, d, _, _, _ = compute_point(
                M, alpha, transition, filepath,
                h_peak_func, tau_func, det, f_band
            )
            f_arr.append(f);  d_arr.append(d);  M_arr.append(M)
            print(f"M = {M:.4e} Msun | f_t = {f:.4e} Hz")
        return np.array(f_arr), np.array(d_arr), np.array(M_arr)

    # Stage 1: coarse
    M_coarse      = np.logspace(np.log10(M_range[0]),
                                np.log10(M_range[1]), n_coarse)
    f_c, d_c, M_c = _sweep(M_coarse)
    fin_c         = np.isfinite(d_c)

    if not fin_c.any():
        raise RuntimeError("No finite points in coarse sweep — check parameters.")

    # Stage 2: dense around f_mech
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

    f_to_logM = interp1d(
        np.log10(f_u), np.log10(M_u),
        kind='linear', bounds_error=False, fill_value=np.nan
    )
    log_M_res = f_to_logM(np.log10(f_mech))

    if np.isfinite(log_M_res):
        M_res_exact = 10.0**log_M_res
        print(f"\nResonant mass: {M_res_exact:.4e} Msun | f_t = f_mech = {f_mech:.4e} Hz")
        f_r, d_r, _, _, _ = compute_point(
            M_res_exact, alpha, transition, filepath,
            h_peak_func, tau_func, det, f_band
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
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_reach(results, alpha, transition_label,
               det=ADMX_EFR, savepath=None):
    """
    Plot the detector distance reach from the output of run_sweep().

    Parameters
    ----------
    results          : dict     — output of run_sweep()
    alpha            : float    — coupling, used only for the legend label
    transition_label : str      — LaTeX label for the transition, e.g. r'|422\rangle \to |322\rangle'
    det              : MagneticWeberBar
    savepath         : str      — file path to save figure, or None to skip saving
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

    # ── Distance reach curve ─────────────────────────────────────────────────
    ax1.loglog(f_all[fin_all], d_all[fin_all],
               color='steelblue', linewidth=2.0,
               label=fr'${transition_label},\ \alpha={alpha}$')

    # ── Resonance point ───────────────────────────────────────────────────────
    if np.isfinite(d_res):
        ax1.scatter([f_res], [d_res],
                    color='firebrick', s=60, zorder=6)

        log_M_val = np.log10(M_res)
        if abs(log_M_val - round(log_M_val)) < 0.15:
            M_label = fr'$M = 10^{{{int(round(log_M_val))}}}\,M_\odot$'
        else:
            M_label = fr'$M = {M_res:.2e}\,M_\odot$'

        ax1.annotate(
            M_label + '\n' + fr'$f_t = f_{{\rm mech}}$',
            xy         = (f_res, d_res),
            xytext     = (f_res * 6.0, d_res * 0.15),
            fontsize   = 9,
            color      = 'firebrick',
            arrowprops = dict(arrowstyle='->', color='firebrick', lw=0.8),
        )

    # ── Vertical line at f_mech ───────────────────────────────────────────────
    ax1.axvline(f_mech, color='firebrick', linewidth=1.0,
                linestyle=':', alpha=0.7)

    # ── Right axis: noise curve ───────────────────────────────────────────────
    ax_noise    = ax1.twinx()
    freqs_noise = np.logspace(2, 8, 2000)
    S_h         = noise_equivalent_strain_broadband(det, freqs_noise)
    ax_noise.loglog(freqs_noise, np.sqrt(S_h),
                    color='gray', linewidth=1.2,
                    linestyle='--', alpha=0.5,
                    label=r'$\sqrt{S_h^{\rm noise}}$')
    ax_noise.set_ylabel(
        r'$\left(S_h^{\rm noise}\right)^{1/2}\ [\mathrm{Hz}^{-1/2}]$',
        fontsize=11, color='gray'
    )
    ax_noise.tick_params(axis='y', labelcolor='gray')

    # ── Top axis: BH mass ─────────────────────────────────────────────────────
    f_fin = f_all[fin_all];  M_fin = M_all[fin_all]
    sort2 = np.argsort(f_fin)
    f_fin = f_fin[sort2];    M_fin = M_fin[sort2]
    _, uniq2 = np.unique(f_fin, return_index=True)
    f_fin    = f_fin[uniq2]; M_fin = M_fin[uniq2]

    log_M_min   = np.ceil(np.log10(M_fin.min()))
    log_M_max   = np.floor(np.log10(M_fin.max()))
    log_M_ticks = np.arange(log_M_min, log_M_max + 1, 1)

    f_tick_pos    = []
    lM_tick_valid = []
    for lM in log_M_ticks:
        M_t   = 10.0**lM
        idx_t = np.argmin(np.abs(M_fin - M_t))
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
        [fr'$10^{{{lM}}}$' for lM in lM_tick_valid],
        fontsize=9
    )
    ax_top.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=12, labelpad=8)

    # ── Labels and legend ─────────────────────────────────────────────────────
    ax1.set_xlabel(r'$f_t\ [\mathrm{Hz}]$', fontsize=13)
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

    # ── Your inputs ───────────────────────────────────────────────────────────
    filepath         = "2. Relativistic Superradiance Rate/Mathematica/SR_n4l2m2_at0.990_aMin0.010_aMax1.200_20260317.dat"
    alpha            = 0.75
    transition       = '4d 3d'
    transition_label = r'|422\rangle \to |322\rangle'
    M_range          = (1e-12, 1e1)   # solar masses
    savepath         = '4. Detector Distance Reach/distance_reach_mass_sweep.pdf'

    # ── Run ───────────────────────────────────────────────────────────────────
    results = run_sweep(
        alpha       = alpha,
        transition  = transition,
        filepath    = filepath,
        h_peak_func = h_peak,
        tau_func    = tau,
        M_range     = M_range,
    )

    plot_reach(
        results          = results,
        alpha            = alpha,
        transition_label = transition_label,
        savepath         = savepath,
    )