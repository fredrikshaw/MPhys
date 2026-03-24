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
# Transition source factory
#
# Returns three callables with the standard interface:
#   freq_func(M_solar)     -> f  [Hz]
#   h_func(f, M_solar)     -> h  [m]    strain at unit distance r = 1 m
#   tau_func(f, M_solar)   -> tau [s]
#
# h_peak for transitions scales as 1/r, so d_max is found by linear inversion.
# ─────────────────────────────────────────────────────────────────────────────

def make_transition_funcs(alpha, transition, filepath):
    """
    Return (freq_func, h_func, tau_func) for a superradiance transition.

    GW frequency is the energy difference between the two levels:
        f_t = omega_t / 2pi

    h_peak ~ 1/r  =>  d_max = h_unit * sqrt(tau / S_h_noise)
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
# Annihilation source factory
#
# For annihilation the peak strain is:
#
#   h_peak = (G_N M^2 Delta_a* / m_a r) * sqrt(8 G_N Gamma_a / r^2 omega_ann)
#          = A / r^2
#
# where A = N_max * sqrt(8 G_N Gamma_a / omega_ann)  has units [m^2]
# and N_max = G_N M^2 Delta_a* / m_a carries dimensions [m^3 s^-2 / (m s^-2)] = [m]
# (since G_N [m^3 kg^-1 s^-2] * M^2 [kg^2] / m_a [kg] / c^2 [m^2 s^-2] = [m])
#
# Because h ~ 1/r^2 rather than 1/r the d_max inversion is different:
#   SNR = (A/r^2) * sqrt(tau/S_h_noise) = rho*
#   r^2 = A * sqrt(tau/S_h_noise) / rho*
#   d_max = sqrt( A * sqrt(tau/S_h_noise) / rho* )
#
# make_annihilation_source returns a single source_func(M_solar) -> (f, A)
# rather than the (freq, h, tau) triple, because the 1/r^2 structure means
# compute_point_ann handles d_max differently to compute_point.
# ─────────────────────────────────────────────────────────────────────────────

def make_annihilation_source(alpha, level, n, l, m, astar_init, debug=False):
    """
    Return source_func(M_solar) -> (f_ann [Hz], A [m^2])
    where h_peak = A / r^2  (r in metres, h dimensionless).

    Also returns tau_func(f_ann, M_solar) -> tau [s].

    Parameters
    ----------
    alpha      : float — dimensionless gravitational coupling
    level      : str   — level string for annihilation rate e.g. '2p'
    n          : int   — principal quantum number
    l          : int   — orbital quantum number
    m          : int   — azimuthal quantum number
    astar_init : float — initial BH spin
    debug      : bool  — if True, print all intermediate values for first point
    """

    G_N_nat      = 6.708e-57    # [eV^-2] — defined at factory scope, visible to all closures
    _debug_done  = [False]

    def source_func(M_solar):
        # ── Natural unit quantities ───────────────────────────────────────────
        r_g          = calc_rg_from_bh_mass(M_solar)        # [eV^-1]
        omega_ann    = calc_omega_ann(r_g, alpha, n)         # [eV]
        ann_rate     = calc_annihilation_rate(
                           level, alpha, omega_ann,
                           G_N=G_N_nat, r_g=r_g
                       )                                     # [eV]
        delta_a_star = calc_delta_astar(astar_init, r_g, alpha, n, m)

        # BH mass in eV: M_eV = r_g / G_N  (from r_g = G_N * M_eV)
        bh_mass_eV   = r_g / G_N_nat                        # [eV]
        n_max        = calc_n_max(bh_mass_eV, delta_a_star, m)   # dimensionless

        # ── GW frequency ─────────────────────────────────────────────────────
        omega_ann_SI = omega_ann * EV_TO_SI                  # [rad/s]
        f_ann        = omega_ann_SI / (2 * np.pi)            # [Hz]

        # ── Source amplitude A ────────────────────────────────────────────────
        # In natural units at r = 1 eV^-1:
        #   h_nat = n_max * sqrt(8 * G_N_nat * ann_rate / omega_ann)
        # Converting to h = A/r^2 with r in metres:
        #   A [m^2] = h_nat * INV_EV_TO_M^2
        h_nat = float(n_max) * np.sqrt(
                    8 * G_N_nat * float(ann_rate) / float(omega_ann)
                )                                            # dimensionless at r=1 eV^-1
        A     = h_nat * INV_EV_TO_M**2                      # [m^2]

        if debug and not _debug_done[0]:
            ann_rate_SI = float(ann_rate) * EV_TO_SI
            print(f"\n{'─'*60}")
            print(f"[ANN DEBUG] M_solar          = {M_solar:.4e} Msun")
            print(f"[ANN DEBUG] r_g              = {r_g:.4e} eV^-1")
            print(f"[ANN DEBUG] r_g (SI)         = {r_g * INV_EV_TO_M:.4e} m")
            print(f"[ANN DEBUG] omega_ann        = {float(omega_ann):.4e} eV")
            print(f"[ANN DEBUG] f_ann            = {f_ann:.4e} Hz")
            print(f"[ANN DEBUG] ann_rate         = {float(ann_rate):.4e} eV")
            print(f"[ANN DEBUG] ann_rate (SI)    = {ann_rate_SI:.4e} s^-1")
            print(f"[ANN DEBUG] bh_mass_eV       = {bh_mass_eV:.4e} eV")
            print(f"[ANN DEBUG] delta_a_star     = {delta_a_star:.4e}")
            print(f"[ANN DEBUG] n_max            = {float(n_max):.4e}")
            print(f"[ANN DEBUG] h_nat (r=1eV^-1) = {h_nat:.4e} [dimensionless]")
            print(f"[ANN DEBUG] INV_EV_TO_M      = {INV_EV_TO_M:.4e} m/eV^-1")
            print(f"[ANN DEBUG] A = h_nat*INV^2  = {A:.4e} m^2")
            print(f"[ANN DEBUG] h at 1 kpc       = {A / KPC_TO_M**2:.4e} [dimensionless]")
            print(f"{'─'*60}\n")
            _debug_done[0] = True

        return f_ann, A

    def tau_func(f_ann, M_solar):
        r_g          = calc_rg_from_bh_mass(M_solar)        # [eV^-1]
        omega_ann    = calc_omega_ann(r_g, alpha, n)         # [eV]
        ann_rate     = calc_annihilation_rate(
                           level, alpha, omega_ann,
                           G_N=G_N_nat, r_g=r_g
                       )                                     # [eV]
        # BH mass in eV for calc_n_max
        bh_mass_eV   = r_g / G_N_nat                        # [eV]
        delta_a_star = calc_delta_astar(astar_init, r_g, alpha, n, m)
        n_max        = calc_n_max(bh_mass_eV, delta_a_star, m)   # dimensionless
        tau_eV       = calc_char_t_ann(ann_rate, n_max)      # [eV^-1]
        return float(tau_eV) / EV_TO_SI                      # [s]

    return source_func, tau_func

# ─────────────────────────────────────────────────────────────────────────────
# Core computation — transitions (h ~ 1/r)
# ─────────────────────────────────────────────────────────────────────────────

def compute_point(M_solar, freq_func, h_func, tau_func,
                  det=ADMX_EFR, f_band=(1e2, 1e8)):
    """
    Compute (f, d_max_kpc, h_unit, tau_val, S_h_noise) for one BH mass.
    For transition processes where h_peak ~ 1/r.

    d_max = h_unit * sqrt(tau / S_h_noise)
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
# Core computation — annihilation (h ~ 1/r^2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_point_ann(M_solar, source_func, tau_func,
                      det=ADMX_EFR, f_band=(1e2, 1e8), rho_star=1.0):
    """
    Compute (f, d_max_kpc, A, tau_val, S_h_noise) for one BH mass.
    For annihilation processes where h_peak = A / r^2.

    SNR = (A/r^2) * sqrt(tau/S_h_noise) = rho*
    => d_max = sqrt( A * sqrt(tau/S_h_noise) / rho* )
    """
    try:
        f_ann, A = source_func(M_solar)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if not (f_band[0] <= f_ann <= f_band[1]):
        return f_ann, np.nan, np.nan, np.nan, np.nan

    try:
        tau_val = tau_func(f_ann, M_solar)
    except Exception:
        return f_ann, np.nan, np.nan, np.nan, np.nan

    S_h_noise = noise_equivalent_strain_broadband(
                    det, np.array([f_ann])
                )[0]

    d_max_m_sq = A * np.sqrt(tau_val / S_h_noise) / rho_star

    if not np.isfinite(d_max_m_sq) or d_max_m_sq <= 0:
        return f_ann, np.nan, A, tau_val, S_h_noise

    d_max_kpc = np.sqrt(d_max_m_sq) / KPC_TO_M

    return f_ann, d_max_kpc, A, tau_val, S_h_noise


# ─────────────────────────────────────────────────────────────────────────────
# Sweep — transitions
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(freq_func, h_func, tau_func,
              M_range=(1e-12, 1e1), n_coarse=300, n_dense=800,
              det=ADMX_EFR, f_band=(1e2, 1e8)):
    """
    Two-stage mass sweep for transition processes (h ~ 1/r).
    """

    def _sweep(M_array):
        f_arr, d_arr, M_arr = [], [], []
        for M in M_array:
            f, d, _, _, _ = compute_point(
                M, freq_func, h_func, tau_func, det, f_band
            )
            f_arr.append(f);  d_arr.append(d);  M_arr.append(M)
            # print(f"M = {M:.4e} Msun | f = {f:.4e} Hz")
        return np.array(f_arr), np.array(d_arr), np.array(M_arr)

    M_coarse      = np.logspace(np.log10(M_range[0]),
                                np.log10(M_range[1]), n_coarse)
    f_c, d_c, M_c = _sweep(M_coarse)
    fin_c         = np.isfinite(d_c)

    if not fin_c.any():
        raise RuntimeError("No finite points in coarse sweep.")

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

    f_res_pt = np.nan;  d_res_pt = np.nan;  M_res_pt = np.nan
    sort_idx = np.argsort(f_c[fin_c])
    f_s      = f_c[fin_c][sort_idx];  M_s = M_c[fin_c][sort_idx]
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
            # print(f"\nResonant mass: {M_res_exact:.4e} Msun | f = f_mech = {f_mech:.4e} Hz")
            f_r, d_r, _, _, _ = compute_point(
                M_res_exact, freq_func, h_func, tau_func, det, f_band
            )
            if np.isfinite(d_r):
                f_res_pt = f_r;  d_res_pt = d_r;  M_res_pt = M_res_exact

    f_all = np.concatenate([f_c, f_d])
    d_all = np.concatenate([d_c, d_d])
    M_all = np.concatenate([M_c, M_d])
    sort_idx = np.argsort(f_all)
    f_all    = f_all[sort_idx];  d_all = d_all[sort_idx];  M_all = M_all[sort_idx]
    fin_all  = np.isfinite(d_all) & np.isfinite(f_all) & (f_all > 0)

    return dict(f=f_all, d=d_all, M=M_all, fin=fin_all,
                f_res=f_res_pt, d_res=d_res_pt, M_res=M_res_pt)


# ─────────────────────────────────────────────────────────────────────────────
# Sweep — annihilation
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep_ann(source_func, tau_func,
                  M_range=(1e-12, 1e1), n_coarse=300, n_dense=800,
                  det=ADMX_EFR, f_band=(1e2, 1e8), rho_star=1.0):
    """
    Two-stage mass sweep for annihilation processes (h ~ 1/r^2).
    """

    def _sweep(M_array):
        f_arr, d_arr, M_arr = [], [], []
        for M in M_array:
            f, d, _, _, _ = compute_point_ann(
                M, source_func, tau_func, det, f_band, rho_star
            )
            f_arr.append(f);  d_arr.append(d);  M_arr.append(M)
            # print(f"M = {M:.4e} Msun | f = {f:.4e} Hz")
        return np.array(f_arr), np.array(d_arr), np.array(M_arr)

    M_coarse      = np.logspace(np.log10(M_range[0]),
                                np.log10(M_range[1]), n_coarse)
    f_c, d_c, M_c = _sweep(M_coarse)
    fin_c         = np.isfinite(d_c)

    if not fin_c.any():
        raise RuntimeError("No finite points in annihilation coarse sweep.")

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

    f_res_pt = np.nan;  d_res_pt = np.nan;  M_res_pt = np.nan
    sort_idx = np.argsort(f_c[fin_c])
    f_s      = f_c[fin_c][sort_idx];  M_s = M_c[fin_c][sort_idx]
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
            # print(f"\nResonant mass (ann): {M_res_exact:.4e} Msun | f = f_mech = {f_mech:.4e} Hz")
            f_r, d_r, _, _, _ = compute_point_ann(
                M_res_exact, source_func, tau_func, det, f_band, rho_star
            )
            if np.isfinite(d_r):
                f_res_pt = f_r;  d_res_pt = d_r;  M_res_pt = M_res_exact

    f_all = np.concatenate([f_c, f_d])
    d_all = np.concatenate([d_c, d_d])
    M_all = np.concatenate([M_c, M_d])
    sort_idx = np.argsort(f_all)
    f_all    = f_all[sort_idx];  d_all = d_all[sort_idx];  M_all = M_all[sort_idx]
    fin_all  = np.isfinite(d_all) & np.isfinite(f_all) & (f_all > 0)

    return dict(f=f_all, d=d_all, M=M_all, fin=fin_all,
                f_res=f_res_pt, d_res=d_res_pt, M_res=M_res_pt)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — works for both processes
# ─────────────────────────────────────────────────────────────────────────────

def plot_reach(results, alpha, process_label,
               det=ADMX_EFR, savepath=None):
    """
    Plot the detector distance reach from the output of run_sweep()
    or run_sweep_ann().

    The x-axis runs in increasing BH mass (left to right), which means
    decreasing frequency. Mass is on the bottom axis (log scale) and
    frequency on the top axis. The resonance point is inserted into the
    main dataset so it appears in the connected curve.

    Parameters
    ----------
    results       : dict  — output of run_sweep() or run_sweep_ann()
    alpha         : float — coupling, for legend label
    process_label : str   — LaTeX string e.g. r'|422\rangle \to |322\rangle'
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

    # ── Insert resonance point into the main dataset ──────────────────────────
    # This ensures the resonance spike appears in the connected line rather
    # than as an isolated marker. We insert it at the correct position sorted
    # by mass (equivalently by descending frequency).
    if np.isfinite(d_res) and np.isfinite(M_res):
        f_all   = np.append(f_all,   f_res)
        d_all   = np.append(d_all,   d_res)
        M_all   = np.append(M_all,   M_res)
        # Re-sort by mass ascending (= frequency descending)
        sort_idx = np.argsort(M_all)
        f_all    = f_all[sort_idx]
        d_all    = d_all[sort_idx]
        M_all    = M_all[sort_idx]
        fin_all  = np.isfinite(d_all) & np.isfinite(f_all) & (f_all > 0)

    fig, ax1 = plt.subplots(figsize=(4, 3.5))
    fig.subplots_adjust(top=0.80)

    # ── Main distance reach curve — plotted vs MASS on bottom axis ───────────
    # Since f ~ 1/M, plotting vs M with log scale and inverted direction
    # means we plot M on x and d on y, with M increasing left to right.
    M_fin = M_all[fin_all]
    d_fin = d_all[fin_all]
    f_fin = f_all[fin_all]

    # Sort by mass ascending for correct line connection
    sort_m   = np.argsort(M_fin)
    M_plot   = M_fin[sort_m]
    d_plot   = d_fin[sort_m]
    f_plot   = f_fin[sort_m]

    ax1.loglog(M_plot, d_plot,
               color='steelblue', linewidth=2.0,
               label=fr'${process_label},\ \alpha={alpha}$')

    # ── Mark the resonance point on the curve ─────────────────────────────────
    if np.isfinite(d_res) and np.isfinite(M_res):
        ax1.scatter([M_res], [d_res],
                    color='firebrick', s=60, zorder=6,
                    label=fr'$f = f_{{\rm mech}}$')

        log_M_val = np.log10(M_res)
        if abs(log_M_val - round(log_M_val)) < 0.15:
            M_label = fr'$M = 10^{{{int(round(log_M_val))}}}\,M_\odot$'
        else:
            M_label = fr'$M = {M_res:.2e}\,M_\odot$'

        ax1.annotate(
            M_label + '\n' + fr'$f = f_{{\rm mech}}$',
            xy         = (M_res, d_res),
            xytext     = (M_res * 4.0, d_res * 0.15),
            fontsize   = 9, color='firebrick',
            arrowprops = dict(arrowstyle='->', color='firebrick', lw=0.8),
        )

    # ── Vertical line at the resonant mass ───────────────────────────────────
    if np.isfinite(M_res):
        ax1.axvline(M_res, color='firebrick', linewidth=1.0,
                    linestyle=':', alpha=0.7)

    # ── Right axis: noise curve vs mass ───────────────────────────────────────
    # The noise curve is defined vs frequency, so we convert a frequency grid
    # to an equivalent mass grid using the monotone f <-> M relationship
    # already established from the sweep data.
    ax_noise = ax1.twinx()

    # Build a monotone interpolator from f -> M using the sorted finite data
    if len(M_plot) >= 2:
        # f decreases as M increases — use unique M values
        _, uniq_m = np.unique(M_plot, return_index=True)
        M_uniq    = M_plot[uniq_m]
        f_uniq    = f_plot[uniq_m]

        f_to_M    = interp1d(
            np.log10(f_uniq[::-1]),   # f decreasing -> M increasing, so reverse
            np.log10(M_uniq[::-1]),
            kind='linear', bounds_error=False, fill_value=np.nan
        )

        freqs_noise = np.logspace(2, 8, 2000)
        S_h         = noise_equivalent_strain_broadband(det, freqs_noise)

        # Convert noise frequencies to masses
        log_M_noise = f_to_M(np.log10(freqs_noise))
        valid_noise = np.isfinite(log_M_noise)
        M_noise     = 10.0**log_M_noise[valid_noise]
        Sn_plot     = np.sqrt(S_h[valid_noise])

        # Sort by mass for correct plotting
        sort_n  = np.argsort(M_noise)
        ax_noise.loglog(M_noise[sort_n], Sn_plot[sort_n],
                        color='gray', linewidth=1.2,
                        linestyle='--', alpha=0.5,
                        label=r'$\sqrt{S_h^{\rm noise}}$')

    ax_noise.set_ylabel(
        r'$\left(S_h^{\rm noise}\right)^{1/2}\ [\mathrm{Hz}^{-1/2}]$',
        fontsize=11, color='gray'
    )
    ax_noise.tick_params(axis='y', labelcolor='gray')

    # ── Top axis: frequency ───────────────────────────────────────────────────
    # Since f ~ 1/M, the top axis runs in decreasing frequency as M increases.
    # We place frequency ticks at round powers of 10 within the plotted range.
    ax_top = ax1.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax1.get_xlim())   # same mass limits

    # Find which masses correspond to round decade frequencies in the band
    f_min_plot = f_plot.min()
    f_max_plot = f_plot.max()
    log_f_min  = np.ceil(np.log10(f_min_plot))
    log_f_max  = np.floor(np.log10(f_max_plot))
    log_f_ticks = np.arange(log_f_min, log_f_max + 1, 1)

    M_tick_pos    = []
    lf_tick_valid = []

    if len(M_plot) >= 2:
        M_to_logf = interp1d(
            np.log10(M_uniq),
            np.log10(f_uniq),
            kind='linear', bounds_error=False, fill_value=np.nan
        )
        xl = ax1.get_xlim()
        for lf in log_f_ticks:
            # Find which mass gives this frequency
            # Since f ~ 1/M, invert: find M from f
            log_M_at_f = f_to_M(lf)
            if np.isfinite(log_M_at_f):
                M_at_f = 10.0**log_M_at_f
                if xl[0] <= M_at_f <= xl[1]:
                    M_tick_pos.append(M_at_f)
                    lf_tick_valid.append(int(lf))

    ax_top.set_xticks(M_tick_pos)
    ax_top.set_xticklabels(
        [fr'$10^{{{lf}}}$' for lf in lf_tick_valid], fontsize=9
    )
    ax_top.set_xlabel(r'$f\ [\mathrm{Hz}]$', fontsize=12, labelpad=8)

    # ── Bottom axis labels ────────────────────────────────────────────────────
    ax1.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=13)
    ax1.set_ylabel(r'$d_{\rm max}\ [\mathrm{kpc}]$', fontsize=13)
    ax1.grid(True, which='both', alpha=0.3)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax_noise.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2,
               fontsize=10, loc='upper left', frameon=False)

    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')

    plt.show()
    return fig, ax1

# ─────────────────────────────────────────────────────────────────────────────
# Main — edit only this block
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Common parameters ─────────────────────────────────────────────────────
    alpha   = 0.01
    M_range = (1e-12, 1e4)   # solar masses
    savepath= '4. Detector Distance Reach/distance_reach_mass_sweep.pdf'

    # =========================================================================
    # OPTION A: Transition process
    # =========================================================================
    filepath        = "2. Relativistic Superradiance Rate/Mathematica/SR_n2l1m1_at0.990_aMin0.010_aMax0.500_20260310.dat"
    transition      = '3p 2p'
    process_label_t = r'|422\rangle \to |322\rangle\ \text{(transition)}'

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
    level           = '2p'
    n, l, m         = 2, 1, 1
    astar_init      = 0.99
    process_label_a = r'|211\rangle\ \text{(annihilation)}'

    source_func_a, tau_func_ann = make_annihilation_source(
        alpha      = alpha,
        level      = level,
        n          = n,
        l          = l,
        m          = m,
        astar_init = astar_init,
        debug      = True,   # prints full diagnostics for the first mass point
    )

    results_a = run_sweep_ann(
        source_func = source_func_a,
        tau_func    = tau_func_ann,
        M_range     = M_range,
    )

    plot_reach(
        results       = results_a,
        alpha         = alpha,
        process_label = process_label_a,
        savepath      = savepath.replace('.pdf', '_ann.pdf'),
    )