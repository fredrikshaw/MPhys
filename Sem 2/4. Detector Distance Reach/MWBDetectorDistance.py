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


def check_required_files(verbose=False):
    """Verify required files and modules are accessible for this script."""
    checks = []

    # folder paths referenced in this module
    checks.append((script_dir, "ParamCalculator script directory"))
    checks.append((relativistic_dir, "SuperradianceRate directory"))

    # module paths for explicit local dependencies
    for module_name in ["MagneticWeberBar", "ParamCalculator", "SuperradianceRateCF"]:
        try:
            module = __import__(module_name)
            module_path = Path(getattr(module, "__file__", ""))
            checks.append((module_path, f"Python module {module_name}"))
        except Exception as ex:
            checks.append((Path("<missing>"), f"Python module {module_name} (import failed: {ex})"))

    all_ok = True
    for path_obj, reason in checks:
        if not path_obj or not path_obj.exists():
            print(f"[FILE VALIDATION] MISSING: {reason}: {path_obj}")
            all_ok = False
        elif verbose:
            print(f"[FILE VALIDATION] OK: {reason}: {path_obj}")
    return all_ok

# Run validation at import-time once
_required_files_ok = check_required_files(verbose=False)

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
                  det=ADMX_EFR, f_band=(1e2, 1e8), rho_star=1.0):
    """
    Compute (f, d_max_kpc, h_unit, tau_val, S_h_noise) for one BH mass.
    For transition processes where h_peak ~ 1/r.

    d_max = h_unit * sqrt(tau / S_h_noise)
    """
    try:
        f_t = freq_func(M_solar)
    except Exception as exc:
        # print(f"[VALIDATION] Rejected point: M={M_solar} (freq_func failed: {exc})")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if not (f_band[0] <= f_t <= f_band[1]):
        # print(f"[VALIDATION] Rejected point: M={M_solar} f={f_t:.3e} (outside f_band {f_band})")
        return f_t, np.nan, np.nan, np.nan, np.nan

    try:
        h_unit = h_func(f_t, M_solar)
    except Exception as exc:
        # print(f"[VALIDATION] Rejected point: M={M_solar} f={f_t:.3e} (h_func failed: {exc})")
        return f_t, np.nan, np.nan, np.nan, np.nan

    try:
        tau_val = tau_func(f_t, M_solar)
    except Exception as exc:
        # print(f"[VALIDATION] Rejected point: M={M_solar} f={f_t:.3e} (tau_func failed: {exc})")
        return f_t, np.nan, np.nan, np.nan, np.nan

    S_h_noise = noise_equivalent_strain_broadband(
                    det, np.array([f_t])
                )[0]

    d_max_m   = h_unit * np.sqrt(tau_val / S_h_noise)
    d_max_kpc = d_max_m / KPC_TO_M

    if not np.isfinite(d_max_kpc) or d_max_kpc <= 0:
        # print(f"[VALIDATION] Rejected point: M={M_solar} f={f_t:.3e} d_max_kpc={d_max_kpc} (invalid distance)")
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
    except Exception as exc:
        # print(f"[VALIDATION] Rejected ann point: M={M_solar} (source_func failed: {exc})")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if not (f_band[0] <= f_ann <= f_band[1]):
        # print(f"[VALIDATION] Rejected ann point: M={M_solar} f={f_ann:.3e} (outside f_band {f_band})")
        return f_ann, np.nan, np.nan, np.nan, np.nan

    try:
        tau_val = tau_func(f_ann, M_solar)
    except Exception as exc:
        # print(f"[VALIDATION] Rejected ann point: M={M_solar} f={f_ann:.3e} (tau_func failed: {exc})")
        return f_ann, np.nan, np.nan, np.nan, np.nan

    S_h_noise = noise_equivalent_strain_broadband(
                    det, np.array([f_ann])
                )[0]

    d_max_m_sq = A * np.sqrt(tau_val / S_h_noise) / rho_star

    if not np.isfinite(d_max_m_sq) or d_max_m_sq <= 0:
        # print(f"[VALIDATION] Rejected ann point: M={M_solar} f={f_ann:.3e} d_max_m_sq={d_max_m_sq} (invalid distance)")
        return f_ann, np.nan, A, tau_val, S_h_noise

    d_max_kpc = np.sqrt(d_max_m_sq) / KPC_TO_M

    return f_ann, d_max_kpc, A, tau_val, S_h_noise


# ─────────────────────────────────────────────────────────────────────────────
# Sweep — transitions
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(freq_func, h_func, tau_func,
              M_range=(1e-12, 1e1), n_coarse=300, n_dense=800,
              det1=ADMX_EFR, det2=DMRADIO_GUT, f_band=(1e2, 1e8), rho_star=1.0):
    """
    Two-stage mass sweep for transition processes (h ~ 1/r).
    Returns results for both detectors.
    """
    
    def _sweep(M_array, det):
        f_arr, d_arr, M_arr = [], [], []
        for M in M_array:
            f, d, _, _, _ = compute_point(
                M, freq_func, h_func, tau_func, det, f_band, rho_star
            )
            f_arr.append(f);  d_arr.append(d);  M_arr.append(M)
        return np.array(f_arr), np.array(d_arr), np.array(M_arr)

    M_coarse = np.logspace(np.log10(M_range[0]), np.log10(M_range[1]), n_coarse)
    
    # Get results for both detectors
    f_c1, d_c1, M_c1 = _sweep(M_coarse, det1)
    f_c2, d_c2, M_c2 = _sweep(M_coarse, det2)
    
    # Find resonant masses for both detectors
    f_mech1 = det1.f_mech
    f_mech2 = det2.f_mech
    
    # Process results for detector 1 - find region around resonance
    fin_c1 = np.isfinite(d_c1)
    M_res1, f_r1, d_r1 = np.nan, np.nan, np.nan
    d_fine1 = None
    M_fine1 = None
    
    if fin_c1.any():
        # Find index closest to mechanical frequency
        idx_closest = np.argmin(np.abs(f_c1[fin_c1] - f_mech1))
        M_guess = M_c1[fin_c1][idx_closest]
        
        # Do dense sweep around the guess
        M_dense = np.logspace(np.log10(M_guess/10), np.log10(M_guess*10), n_dense)
        f_dense, d_dense, _ = _sweep(M_dense, det1)
        
        # Store dense results for plotting
        valid_dense = np.isfinite(d_dense)
        if valid_dense.any():
            M_fine1 = M_dense[valid_dense]
            d_fine1 = d_dense[valid_dense]
            
            # Find exact resonance from dense sweep
            idx_res = np.argmax(d_fine1)
            M_res1 = M_fine1[idx_res]
            f_r1 = f_dense[valid_dense][idx_res]
            d_r1 = d_fine1[idx_res]
    
    # Process results for detector 2
    fin_c2 = np.isfinite(d_c2)
    M_res2, f_r2, d_r2 = np.nan, np.nan, np.nan
    d_fine2 = None
    M_fine2 = None
    
    if fin_c2.any():
        # Find index closest to mechanical frequency
        idx_closest = np.argmin(np.abs(f_c2[fin_c2] - f_mech2))
        M_guess = M_c2[fin_c2][idx_closest]
        
        # Do dense sweep around the guess
        M_dense = np.logspace(np.log10(M_guess/10), np.log10(M_guess*10), n_dense)
        f_dense, d_dense, _ = _sweep(M_dense, det2)
        
        # Store dense results for plotting
        valid_dense = np.isfinite(d_dense)
        if valid_dense.any():
            M_fine2 = M_dense[valid_dense]
            d_fine2 = d_dense[valid_dense]
            
            # Find exact resonance from dense sweep
            idx_res = np.argmax(d_fine2)
            M_res2 = M_fine2[idx_res]
            f_r2 = f_dense[valid_dense][idx_res]
            d_r2 = d_fine2[idx_res]
    
    # Combine coarse and dense results for plotting
    # For detector 1
    valid_c1 = np.isfinite(d_c1)
    if M_fine1 is not None:
        M_combined1 = np.concatenate([M_c1[valid_c1], M_fine1])
        d_combined1 = np.concatenate([d_c1[valid_c1], d_fine1])
        # Sort by mass
        sort_idx = np.argsort(M_combined1)
        M_combined1 = M_combined1[sort_idx]
        d_combined1 = d_combined1[sort_idx]
    else:
        M_combined1 = M_c1[valid_c1]
        d_combined1 = d_c1[valid_c1]
    
    # For detector 2
    valid_c2 = np.isfinite(d_c2)
    if M_fine2 is not None:
        M_combined2 = np.concatenate([M_c2[valid_c2], M_fine2])
        d_combined2 = np.concatenate([d_c2[valid_c2], d_fine2])
        # Sort by mass
        sort_idx = np.argsort(M_combined2)
        M_combined2 = M_combined2[sort_idx]
        d_combined2 = d_combined2[sort_idx]
    else:
        M_combined2 = M_c2[valid_c2]
        d_combined2 = d_c2[valid_c2]
    
    return {
        'det1': {
            'name': 'ADMX-EFR',
            'f_mech': f_mech1,
            'f': f_c1[valid_c1],
            'd': d_c1[valid_c1],
            'M': M_c1[valid_c1],
            'f_combined': f_c1[valid_c1],  # Keep for compatibility
            'd_combined': d_combined1,
            'M_combined': M_combined1,
            'f_res': f_r1,
            'd_res': d_r1,
            'M_res': M_res1
        },
        'det2': {
            'name': 'DMRadio-GUT',
            'f_mech': f_mech2,
            'f': f_c2[valid_c2],
            'd': d_c2[valid_c2],
            'M': M_c2[valid_c2],
            'f_combined': f_c2[valid_c2],  # Keep for compatibility
            'd_combined': d_combined2,
            'M_combined': M_combined2,
            'f_res': f_r2,
            'd_res': d_r2,
            'M_res': M_res2
        }
    }

# ─────────────────────────────────────────────────────────────────────────────
# Sweep — annihilation
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep_ann(source_func, tau_func,
                  M_range=(1e-12, 1e1), n_coarse=300, n_dense=800,
                  det1=ADMX_EFR, det2=DMRADIO_GUT, f_band=(1e2, 1e8), rho_star=1.0):
    """
    Two-stage mass sweep for annihilation processes (h ~ 1/r^2).
    Returns results for both detectors.
    """
    
    def _sweep(M_array, det):
        f_arr, d_arr, M_arr = [], [], []
        for M in M_array:
            f, d, _, _, _ = compute_point_ann(
                M, source_func, tau_func, det, f_band, rho_star
            )
            f_arr.append(f);  d_arr.append(d);  M_arr.append(M)
        return np.array(f_arr), np.array(d_arr), np.array(M_arr)

    M_coarse = np.logspace(np.log10(M_range[0]), np.log10(M_range[1]), n_coarse)
    
    # Get results for both detectors
    f_c1, d_c1, M_c1 = _sweep(M_coarse, det1)
    f_c2, d_c2, M_c2 = _sweep(M_coarse, det2)
    
    # Find resonant masses for both detectors
    f_mech1 = det1.f_mech
    f_mech2 = det2.f_mech
    
    # Process results for detector 1
    fin_c1 = np.isfinite(d_c1)
    M_res1, f_r1, d_r1 = np.nan, np.nan, np.nan
    d_fine1 = None
    M_fine1 = None
    
    if fin_c1.any():
        # Find index closest to mechanical frequency
        idx_closest = np.argmin(np.abs(f_c1[fin_c1] - f_mech1))
        M_guess = M_c1[fin_c1][idx_closest]
        
        # Do dense sweep around the guess
        M_dense = np.logspace(np.log10(M_guess/10), np.log10(M_guess*10), n_dense)
        f_dense, d_dense, _ = _sweep(M_dense, det1)
        
        # Store dense results for plotting
        valid_dense = np.isfinite(d_dense)
        if valid_dense.any():
            M_fine1 = M_dense[valid_dense]
            d_fine1 = d_dense[valid_dense]
            
            # Find exact resonance from dense sweep
            idx_res = np.argmax(d_fine1)
            M_res1 = M_fine1[idx_res]
            f_r1 = f_dense[valid_dense][idx_res]
            d_r1 = d_fine1[idx_res]
    
    # Process results for detector 2
    fin_c2 = np.isfinite(d_c2)
    M_res2, f_r2, d_r2 = np.nan, np.nan, np.nan
    d_fine2 = None
    M_fine2 = None
    
    if fin_c2.any():
        # Find index closest to mechanical frequency
        idx_closest = np.argmin(np.abs(f_c2[fin_c2] - f_mech2))
        M_guess = M_c2[fin_c2][idx_closest]
        
        # Do dense sweep around the guess
        M_dense = np.logspace(np.log10(M_guess/10), np.log10(M_guess*10), n_dense)
        f_dense, d_dense, _ = _sweep(M_dense, det2)
        
        # Store dense results for plotting
        valid_dense = np.isfinite(d_dense)
        if valid_dense.any():
            M_fine2 = M_dense[valid_dense]
            d_fine2 = d_dense[valid_dense]
            
            # Find exact resonance from dense sweep
            idx_res = np.argmax(d_fine2)
            M_res2 = M_fine2[idx_res]
            f_r2 = f_dense[valid_dense][idx_res]
            d_r2 = d_fine2[idx_res]
    
    # Combine coarse and dense results for plotting
    valid_c1 = np.isfinite(d_c1)
    if M_fine1 is not None:
        M_combined1 = np.concatenate([M_c1[valid_c1], M_fine1])
        d_combined1 = np.concatenate([d_c1[valid_c1], d_fine1])
        # Sort by mass
        sort_idx = np.argsort(M_combined1)
        M_combined1 = M_combined1[sort_idx]
        d_combined1 = d_combined1[sort_idx]
    else:
        M_combined1 = M_c1[valid_c1]
        d_combined1 = d_c1[valid_c1]
    
    valid_c2 = np.isfinite(d_c2)
    if M_fine2 is not None:
        M_combined2 = np.concatenate([M_c2[valid_c2], M_fine2])
        d_combined2 = np.concatenate([d_c2[valid_c2], d_fine2])
        # Sort by mass
        sort_idx = np.argsort(M_combined2)
        M_combined2 = M_combined2[sort_idx]
        d_combined2 = d_combined2[sort_idx]
    else:
        M_combined2 = M_c2[valid_c2]
        d_combined2 = d_c2[valid_c2]
    
    return {
        'det1': {
            'name': 'ADMX-EFR',
            'f_mech': f_mech1,
            'f': f_c1[valid_c1],
            'd': d_c1[valid_c1],
            'M': M_c1[valid_c1],
            'd_combined': d_combined1,
            'M_combined': M_combined1,
            'f_res': f_r1,
            'd_res': d_r1,
            'M_res': M_res1
        },
        'det2': {
            'name': 'DMRadio-GUT',
            'f_mech': f_mech2,
            'f': f_c2[valid_c2],
            'd': d_c2[valid_c2],
            'M': M_c2[valid_c2],
            'd_combined': d_combined2,
            'M_combined': M_combined2,
            'f_res': f_r2,
            'd_res': d_r2,
            'M_res': M_res2
        }
    }

# ─────────────────────────────────────────────────────────────────────────────
# Plotting — works for both processes
# ─────────────────────────────────────────────────────────────────────────────

def plot_reach(results, alpha, process_label,
               savepath=None):
    """
    Plot the detector distance reach for both ADMX-EFR and DMRadio-GUT.
    """
    plt.rcParams.update({
        "text.usetex"         : True,
        "font.family"         : "serif",
        "font.serif"          : ["Computer Modern Roman"],
        "text.latex.preamble" : r"\usepackage{amsmath}"
    })

    fig, ax1 = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(top=0.80)

    colors = {'ADMX-EFR': 'steelblue', 'DMRadio-GUT': 'darkorange'}
    linestyles = {'ADMX-EFR': '-', 'DMRadio-GUT': '--'}
    
    # Plot distance reach for both detectors
    for det_key in ['det1', 'det2']:
        det_data = results[det_key]
        
        # Use combined data for smoother plot around resonance
        M_plot = det_data['M_combined']
        d_plot = det_data['d_combined']
        
        if len(M_plot) == 0:
            continue
        
        # Sort by mass
        sort_idx = np.argsort(M_plot)
        M_plot = M_plot[sort_idx]
        d_plot = d_plot[sort_idx]
        
        # Remove any duplicate masses that might cause plotting artifacts
        _, unique_idx = np.unique(M_plot, return_index=True)
        M_plot = M_plot[unique_idx]
        d_plot = d_plot[unique_idx]
        
        # Create label
        label = f"{det_data['name']}: {process_label}, $\\alpha={alpha}$"
        
        # Plot distance reach
        ax1.loglog(M_plot, d_plot,
                   color=colors[det_data['name']],
                   linewidth=2.0,
                   linestyle=linestyles[det_data['name']],
                   label=label)
        
        # Mark resonance point if exists
        if np.isfinite(det_data['d_res']) and np.isfinite(det_data['M_res']):
            ax1.scatter([det_data['M_res']], [det_data['d_res']],
                        color=colors[det_data['name']], s=80, zorder=6,
                        marker='o' if det_key == 'det1' else 's',
                        edgecolors='black', linewidth=0.5)
            
            # Add annotation for the resonance
            if det_key == 'det1':
                log_M_val = np.log10(det_data['M_res'])
                if abs(log_M_val - round(log_M_val)) < 0.15:
                    M_label = f'$M = 10^{{{int(round(log_M_val))}}}\\,M_\\odot$'
                else:
                    M_label = f'$M = {det_data["M_res"]:.2e}\\,M_\\odot$'
                
                ax1.annotate(
                    M_label + '\n' + '$f = f_{\\rm mech}^{\\rm ADMX}$',
                    xy         = (det_data['M_res'], det_data['d_res']),
                    xytext     = (det_data['M_res'] * 4.0, det_data['d_res'] * 0.15),
                    fontsize   = 8, color=colors[det_data['name']],
                    arrowprops = dict(arrowstyle='->', color=colors[det_data['name']], lw=0.8),
                )
            elif det_key == 'det2' and np.isfinite(det_data['d_res']):
                ax1.annotate(
                    '$f = f_{\\rm mech}^{\\rm DMRadio}$',
                    xy         = (det_data['M_res'], det_data['d_res']),
                    xytext     = (det_data['M_res'] * 0.3, det_data['d_res'] * 0.3),
                    fontsize   = 8, color=colors[det_data['name']],
                    arrowprops = dict(arrowstyle='->', color=colors[det_data['name']], lw=0.8),
                )
            
            # Add vertical line at resonance
            ax1.axvline(det_data['M_res'], color=colors[det_data['name']], 
                       linewidth=1.0, linestyle=':', alpha=0.5, zorder=1)

    # Add noise curves on right axis - now using a smooth interpolation
    ax_noise = ax1.twinx()
    
    # Create a smooth frequency-to-mass mapping using the coarse data only
    det1_data = results['det1']
    
    # Use the coarse data (not combined) for mapping - this gives smooth interpolation
    valid_coarse = np.isfinite(det1_data['d']) & np.isfinite(det1_data['M']) & (det1_data['d'] > 0)
    
    if valid_coarse.any():
        M_coarse = det1_data['M'][valid_coarse]
        f_coarse = det1_data['f'][valid_coarse]
        
        # Sort by mass
        sort_idx = np.argsort(M_coarse)
        M_coarse_sorted = M_coarse[sort_idx]
        f_coarse_sorted = f_coarse[sort_idx]
        
        # Remove duplicates for interpolation
        _, unique_idx = np.unique(M_coarse_sorted, return_index=True)
        M_coarse_unique = M_coarse_sorted[unique_idx]
        f_coarse_unique = f_coarse_sorted[unique_idx]
        
        if len(M_coarse_unique) >= 2:
            from scipy.interpolate import interp1d
            # Create mapping from frequency to mass (in log space)
            # Need to ensure frequencies are monotonic
            sort_f_idx = np.argsort(f_coarse_unique)
            f_sorted = f_coarse_unique[sort_f_idx]
            M_sorted = M_coarse_unique[sort_f_idx]
            
            # Remove any duplicates in frequency
            _, f_unique_idx = np.unique(f_sorted, return_index=True)
            f_for_interp = f_sorted[f_unique_idx]
            M_for_interp = M_sorted[f_unique_idx]
            
            if len(f_for_interp) >= 2:
                f_to_M = interp1d(np.log10(f_for_interp), np.log10(M_for_interp),
                                 kind='linear', bounds_error=False, fill_value=np.nan)
                
                # Create smooth frequency array for noise curves
                freqs_noise = np.logspace(2, 8, 5000)  # Increased points for smoothness
                
                # Get noise for both detectors
                from MagneticWeberBar import noise_equivalent_strain_broadband, ADMX_EFR, DMRADIO_GUT
                S_h_efr = noise_equivalent_strain_broadband(ADMX_EFR, freqs_noise)
                S_h_dmradio = noise_equivalent_strain_broadband(DMRADIO_GUT, freqs_noise)
                
                # Convert to masses using interpolation
                log_M_noise = f_to_M(np.log10(freqs_noise))
                valid_noise = np.isfinite(log_M_noise)
                
                if valid_noise.any():
                    M_noise = 10.0**log_M_noise[valid_noise]
                    freqs_valid = freqs_noise[valid_noise]
                    
                    # Get noise values only where conversion is valid
                    Sn_efr_valid = np.sqrt(S_h_efr[valid_noise])
                    Sn_dmradio_valid = np.sqrt(S_h_dmradio[valid_noise])
                    
                    # Sort by mass for clean plotting
                    sort_m_idx = np.argsort(M_noise)
                    M_noise_sorted = M_noise[sort_m_idx]
                    Sn_efr_sorted = Sn_efr_valid[sort_m_idx]
                    Sn_dmradio_sorted = Sn_dmradio_valid[sort_m_idx]
                    
                    # Plot smooth noise curves
                    ax_noise.loglog(M_noise_sorted, Sn_efr_sorted,
                                   color='gray', linewidth=1.2, linestyle='--', alpha=0.7,
                                   label=r'ADMX-EFR $\sqrt{S_h^{\rm noise}}$')
                    ax_noise.loglog(M_noise_sorted, Sn_dmradio_sorted,
                                   color='darkgray', linewidth=1.2, linestyle=':', alpha=0.7,
                                   label=r'DMRadio-GUT $\sqrt{S_h^{\rm noise}}$')
    
    ax_noise.set_ylabel(r'$\left(S_h^{\rm noise}\right)^{1/2}\ [\mathrm{Hz}^{-1/2}]$',
                        fontsize=11, color='gray')
    ax_noise.tick_params(axis='y', labelcolor='gray')
    
    # Bottom axis labels
    ax1.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=13)
    ax1.set_ylabel(r'$d_{\rm max}\ [\mathrm{kpc}]$', fontsize=13)
    # ax1.grid(True, which='both', alpha=0.3) # no
    
    # Combine legends
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax_noise.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2,
              fontsize=8, loc='upper left', frameon=False)
    
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
    M_range = (1e-12, 1e4)
    savepath= '4. Detector Distance Reach/distance_reach_mass_sweep.pdf'

    # =========================================================================
    # OPTION A: Transition process
    # =========================================================================
    filepath = "2. Relativistic Superradiance Rate/Mathematica/SR_n2l1m1_at0.990_aMin0.010_aMax0.500_20260310.dat"
    transition = '3p 2p'
    # Use simpler label without \text command
    process_label_t = r'$|422\rangle \to |322\rangle$ (transition)'

    freq_func_t, h_func_t, tau_func_t = make_transition_funcs(
        alpha=alpha, transition=transition, filepath=filepath,
    )

    results_t = run_sweep(
        freq_func=freq_func_t, h_func=h_func_t, tau_func=tau_func_t,
        M_range=M_range, rho_star=1.0,
    )

    plot_reach(results_t, alpha, process_label_t, savepath=savepath)

    # =========================================================================
    # OPTION B: Annihilation process
    # =========================================================================
    level = '2p'
    n, l, m = 2, 1, 1
    astar_init = 0.99
    # Use simpler label without \text command
    process_label_a = r'$|211\rangle$ (annihilation)'

    source_func_a, tau_func_ann = make_annihilation_source(
        alpha=alpha, level=level, n=n, l=l, m=m,
        astar_init=astar_init, debug=True,
    )

    results_a = run_sweep_ann(
        source_func=source_func_a, tau_func=tau_func_ann,
        M_range=M_range, rho_star=1.0,
    )

    plot_reach(results_a, alpha, process_label_a, 
               savepath=savepath.replace('.pdf', '_ann.pdf'))