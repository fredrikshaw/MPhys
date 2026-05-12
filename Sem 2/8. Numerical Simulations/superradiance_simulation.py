"""
======================================================================
Black Hole Superradiance Simulation — Multi-level (n ≤ 8, m = l)
======================================================================
Simulates the coupled growth of all superradiant axion cloud levels
with m = l (the dominant SR family) up to n = 8, and the resulting
Kerr BH spin-down with shared backreaction across all levels.

Physical inputs are set in solar masses and dimensionless quantities.
All internal computation is in natural units (G = c = ħ = 1, Planck).
Physical units (years, solar masses) are applied only at the plotting stage.

Refactored: simulation run, result storage, plotting, peak analysis,
and data export are separated into dedicated functions.
"""

import os, sys
import csv
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# ── Leaver module import ───────────────────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_LEAVER_DIR = os.path.join(_THIS_DIR, '..', '2. Relativistic Superradiance Rate')
sys.path.insert(0, _LEAVER_DIR if os.path.isdir(_LEAVER_DIR) else _THIS_DIR)

from leaver_superradiance import hydrogen_gamma

# ── Import ParamCalculator ────────────────────────────────────────────────────
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
    calc_transition_rate,
    calc_annihilation_rate,
    calc_omega_ann
)

# ══════════════════════════════════════════════════════════════════════
# Physical constants  (unit conversion at plot stage only)
# ══════════════════════════════════════════════════════════════════════

M_SUN_KG       = 1.989e30    # kg
M_PL_KG        = 2.176e-8    # kg
YEAR_S         = 3.156e7     # s per year
M_SUN_PLANCK   = M_SUN_KG / M_PL_KG    # ≈ 9.137e37  M_Pl per M_sun
GM_SUN_OVER_C3 = 4.927e-6    # s  (GM_sun/c^3, natural BH time unit)

# GW strain unit conversions
L_PL_M         = 1.616e-35   # Planck length [m]
T_PL_S         = 5.391e-44   # Planck time   [s]
KPC_M          = 3.086e19    # 1 kpc [m]
KPC_PLANCK     = KPC_M / L_PL_M        # 1 kpc in Planck lengths ≈ 1.910e54
# GW frequency conversion:  f_GW [Hz] = omega [M_Pl] / (2π × T_PL_S)
OMEGA_PL_TO_HZ = 1.0 / (2.0 * np.pi * T_PL_S)

# Spectroscopic notation for level labels
_L_TO_SPEC = {1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h', 6: 'i', 7: 'k'}

# ══════════════════════════════════════════════════════════════════════
# Horizon / SR condition helpers
# ══════════════════════════════════════════════════════════════════════

def rhat_plus(astar):
    """Dimensionless horizon  r̂₊ = 1 + √(1 − a*²)"""
    return 1.0 + np.sqrt(np.clip(1.0 - astar**2, 0.0, 1.0))

def sr_margin(alpha, astar, m=1):
    """
    Signed SR margin  m·Ω_H − α  where  Ω_H = a*/(2r̂₊).
    Positive → SR active.  Zero → boundary.  Negative → SR off.
    """
    return m * astar / (2.0 * rhat_plus(astar)) - alpha

def irreducible_mass(M, astar):
    """
    Christodoulou-Ruffini irreducible mass of a Kerr BH.
    """
    astar = float(np.clip(astar, 0.0, 0.99999))
    return M * np.sqrt((1.0 + np.sqrt(1.0 - astar**2)) / 2.0)

# ══════════════════════════════════════════════════════════════════════
# Superradiance rate  (hydrogenic / NR approximation)
# ══════════════════════════════════════════════════════════════════════

def gamma_tilde_sr(alpha, astar, n=2, l=1, m=1):
    """
    Dimensionless SR rate  Γ̃_SR = Γ_SR / μ  in the hydrogenic approximation.
    """
    if alpha <= 0.0 or astar <= 0.0:
        return 0.0
    g = hydrogen_gamma(n, l, m, alpha, float(astar))
    return g / alpha if g > 0.0 else 0.0

# ══════════════════════════════════════════════════════════════════════
# Annihilation & transition availability checks (generic)
# ══════════════════════════════════════════════════════════════════════

def _level_string(n, l):
    """(n, l) → spectroscopic string expected by calc_annihilation_rate."""
    spec = _L_TO_SPEC.get(l)
    return f"{n}{spec}" if spec is not None else None

def _check_ann_available(n, l):
    """
    Test whether calc_annihilation_rate has a formula for (n,l).
    """
    lev_str = _level_string(n, l)
    if lev_str is None:
        return False
    try:
        omega_test = calc_omega_ann(r_g=1.0, alpha=0.1, n=n)
        with np.errstate(over='ignore', invalid='ignore'):
            val = calc_annihilation_rate(lev_str, alpha=0.1,
                                         omega=omega_test, G_N=1.0, r_g=1.0)
        return (val is not None and np.isfinite(float(val)) and float(val) >= 0.0)
    except Exception:
        return False

def _check_tr_available(n_i, l_i, n_j, l_j):
    """
    Test whether calc_transition_rate has a formula for (n_i,l_i)→(n_j,l_j).
    """
    key = f"{n_i}{_L_TO_SPEC.get(l_i)} {n_j}{_L_TO_SPEC.get(l_j)}"
    try:
        alpha_test = 0.1
        omega_test = (alpha_test**3 / 2.0) * (1.0 / n_j**2 - 1.0 / n_i**2)
        if omega_test <= 0.0:
            return False
        with np.errstate(over='ignore', invalid='ignore'):
            val = calc_transition_rate(key, alpha_test, omega_test,
                                       G_N=1.0, r_g=1.0)
        return (val is not None and np.isfinite(float(val)) and float(val) >= 0.0)
    except Exception:
        return False

# ══════════════════════════════════════════════════════════════════════
# Data class to hold all simulation results
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SimulationResults:
    # Input parameters
    M_BH_solar: float
    a_star_0: float
    alpha_0: float
    max_n: int

    # Derived constants
    M0: float
    MU: float
    TAU_TO_YR: float

    # Level list
    LEVELS: List[Tuple[int, int, int]]
    _L_ARR: np.ndarray
    _M_ARR: np.ndarray
    N_LEVELS: int

    # Availability dicts
    ann_available: Dict[Tuple[int, int], bool]
    tr_pairs_idx: List[Tuple[int, int, int, int, int, int]]   # (idx_i, idx_j, n_i, l_i, n_j, l_j)

    # Raw ODE output
    tau: np.ndarray
    lnN_all: np.ndarray         # (N_LEVELS, n_time)
    Mtil: np.ndarray            # (n_time,)
    astar: np.ndarray           # (n_time,)

    # Dimensionless normalisation and occupation
    NORM: float                 # = M0^2  (Planck, G_N=1)
    eps_all: np.ndarray         # (N_LEVELS, n_time)  eps_i = N_i / M0^2

    # Derived quantities (computed after integration)
    t_yr: np.ndarray
    N_all: np.ndarray           # (N_LEVELS, n_time)
    alpha: np.ndarray           # (n_time,)
    M_BH: np.ndarray            # (n_time,)
    J_cloud: np.ndarray         # (n_time,)
    M_cloud: np.ndarray         # (n_time,)
    J_BH: np.ndarray            # (n_time,)
    J_total: np.ndarray         # (n_time,)
    dJ_frac: np.ndarray         # (n_time,)
    M_irr: np.ndarray           # (n_time,)
    E_rot: np.ndarray           # (n_time,)
    dM_irr: np.ndarray          # (n_time,)

    # Rates over time
    g_sr_all: np.ndarray        # (N_LEVELS, n_time)
    g_ann_all: np.ndarray       # (N_LEVELS, n_time)

    # GW strain from annihilation
    h_all: np.ndarray           # (N_LEVELS, n_time)   physical h at 1 kpc
    h_dimless_all: np.ndarray   # (N_LEVELS, n_time)   h * r_1kpc / M0  (mass-independent)
    f_gw_hz: np.ndarray         # (N_LEVELS,)
    P_GW_total: np.ndarray      # (n_time,)

    # GW strain from transitions
    g_tr_matrix: np.ndarray     # (n_tr_pairs, n_time)
    h_tr_matrix: np.ndarray     # (n_tr_pairs, n_time)  physical h at 1 kpc
    h_dimless_tr: np.ndarray    # (n_tr_pairs, n_time)  h_tr * r_1kpc / M0
    f_tr_hz: np.ndarray         # (n_tr_pairs,)

    # Summary statistics
    astar_f: float
    t_final_yr: float
    active_levels_t0: int = 0
    dominant_level: Tuple[int, int, int] = field(default_factory=lambda: (2,1,1))


# ══════════════════════════════════════════════════════════════════════
# Main simulation function
# ══════════════════════════════════════════════════════════════════════

def run_simulation(M_BH_solar=1e-11, a_star_0=0.65, alpha_0=0.6,
                   max_n=6, tau_end_factor=None):
    """
    Run the multi-level superradiance simulation.

    Parameters
    ----------
    M_BH_solar : float
        Initial BH mass in solar masses.
    a_star_0 : float
        Initial dimensionless spin.
    alpha_0 : float
        Initial gravitational coupling α₀ = M₀·μ.
    max_n : int
        Maximum principal quantum number n (2 … max_n).
    tau_end_factor : float or None
        Override factor for the integration time. If None, uses heuristic.

    Returns
    -------
    SimulationResults or None
        Object containing all simulation data and metadata.
        Returns None if no level is SR-active at initial parameters.
    """
    # ── Internal derived quantities ──────────────────────────────────
    M0  = M_BH_solar * M_SUN_PLANCK    # Initial BH mass [M_Pl]
    MU  = alpha_0 / M0                 # Axion mass [M_Pl]
    N0  = 1.0                          # Vacuum seed

    # Physical time conversion: t [yr] = τ × TAU_TO_YR
    TAU_TO_YR = M_BH_solar * GM_SUN_OVER_C3 / (alpha_0 * YEAR_S)

    # ── Level list (m = l, 2 ≤ n ≤ max_n) ────────────────────────────
    LEVELS = [(n, l, l) for l in range(1, max_n) for n in range(l+1, max_n+1)]
    N_LEVELS = len(LEVELS)
    _L_ARR = np.array([l for (n, l, m) in LEVELS])
    _M_ARR = np.array([m for (n, l, m) in LEVELS])

    # ── Build availability dictionaries ──────────────────────────────
    # Annihilation
    ann_available = {}
    for (n, l, m) in LEVELS:
        ann_available[(n, l)] = _check_ann_available(n, l)
    ann_have = [(n, l) for (n, l), v in ann_available.items() if v]
    ann_missing = [(n, l) for (n, l), v in ann_available.items() if not v]
    print(f"[annihilation] rates available for {len(ann_have)}/{len(ann_available)} levels")
    if ann_missing:
        print("[annihilation] NOT yet computed: " +
              ", ".join(f"|{n}{_L_TO_SPEC.get(l,'?')}⟩" for n, l in ann_missing))

    # Transition
    tr_available = {}
    for n_i, l_i, m_i in LEVELS:
        for n_j, l_j, m_j in LEVELS:
            if n_i > n_j:
                tr_available[(n_i, l_i, n_j, l_j)] = _check_tr_available(n_i, l_i, n_j, l_j)
    tr_have = [k for k, v in tr_available.items() if v]
    print(f"[transition]   rates available for {len(tr_have)}/{len(tr_available)} pairs")
    if tr_have:
        print("[transition]   active pairs: " +
              ", ".join(f"|{ni}{li}{li}⟩→|{nj}{lj}{lj}⟩" for ni, li, nj, lj in tr_have))

    # Precomputed index lookup for ODE efficiency
    _LEVEL_IDX = {(n, l, m): k for k, (n, l, m) in enumerate(LEVELS)}
    TR_PAIRS_IDX = [
        (_LEVEL_IDX[(n_i, l_i, l_i)], _LEVEL_IDX[(n_j, l_j, l_j)],
         n_i, l_i, n_j, l_j)
        for (n_i, l_i, n_j, l_j), avail in tr_available.items()
        if avail and (n_i, l_i, l_i) in _LEVEL_IDX and (n_j, l_j, l_j) in _LEVEL_IDX
    ]

    # ── Local rate functions (capture alpha_0, M_BH_solar) ───────────
    def gamma_tilde_ann(n, l, Mtil):
        """Dimensionless annihilation rate."""
        if not ann_available.get((n, l), False):
            return 0.0
        alpha_cur = alpha_0 * Mtil
        bh_mass_solar = Mtil * M_BH_solar
        try:
            r_g_ev = calc_rg_from_bh_mass(bh_mass_solar)
            mu_ev = alpha_cur / r_g_ev
            omega_ev = calc_omega_ann(r_g=r_g_ev, alpha=alpha_cur, n=n)
            with np.errstate(over='ignore', invalid='ignore'):
                gamma_a_ev = calc_annihilation_rate(
                    _level_string(n, l), alpha=alpha_cur,
                    omega=omega_ev, G_N=G_N, r_g=r_g_ev)
            val = float(gamma_a_ev)
            if not np.isfinite(val) or val <= 0.0 or mu_ev <= 0.0:
                return 0.0
            return val / mu_ev
        except Exception:
            return 0.0

    def gamma_tilde_tr(n_i, l_i, n_j, l_j, Mtil):
        """Dimensionless transition rate."""
        if not tr_available.get((n_i, l_i, n_j, l_j), False):
            return 0.0
        alpha_cur = alpha_0 * Mtil
        bh_mass_solar = Mtil * M_BH_solar
        try:
            r_g_ev = calc_rg_from_bh_mass(bh_mass_solar)
            mu_ev = alpha_cur / r_g_ev
            omega_ev = (alpha_cur**3 / (2.0 * r_g_ev)) * (1.0 / n_j**2 - 1.0 / n_i**2)
            if omega_ev <= 0.0 or mu_ev <= 0.0:
                return 0.0
            key = f"{n_i}{_L_TO_SPEC.get(l_i)} {n_j}{_L_TO_SPEC.get(l_j)}"
            with np.errstate(over='ignore', invalid='ignore'):
                gamma_tr_ev = calc_transition_rate(
                    key, alpha_cur, omega_ev, G_N=G_N, r_g=r_g_ev)
            val = float(gamma_tr_ev)
            if not np.isfinite(val) or val <= 0.0:
                return 0.0
            return val / mu_ev
        except Exception:
            return 0.0

    # ── ODE system ───────────────────────────────────────────────────
    def odes(tau, y):
        """
        State: y = [ln_eps_0, ..., ln_eps_{K-1}, Mtil, astar]
        where  eps_i = N_i / NORM = N_i / M0^2  (dimensionless occupation).

        The BH equations simplify because NORM cancels:
            dMtil/dtau  = -alpha_0 * dot(g_sr, eps)
            dastar/dtau = dot(g_sr * eps, 2*alpha*astar - m_i) / Mtil^2
        """
        ln_eps_arr = y[:N_LEVELS]
        Mtil       = max(y[N_LEVELS],     1e-9)
        astar      = float(np.clip(y[N_LEVELS + 1], 0.0, 0.99999))

        # Prevent overflow when converting ln(eps) -> eps and forming N = NORM * eps.
        # Compute a safe cap for ln(eps) so that N = exp(log(NORM) + ln_eps) stays finite.
        float_max_log = np.log(np.finfo(float).max)
        # safety margin to avoid hitting the absolute max during arithmetic
        safety_margin = 5.0
        try:
            log_NORM = float(np.log(NORM))
        except Exception:
            log_NORM = float_max_log - 100.0
        ln_eps_cap = min(700.0, float_max_log - log_NORM - safety_margin)

        ln_eps_clipped = np.clip(ln_eps_arr, None, ln_eps_cap)
        # compute ln(N) directly and clip before exponentiating to avoid overflow
        ln_N = log_NORM + ln_eps_clipped
        ln_N_cap = float_max_log - safety_margin
        ln_N_clipped = np.minimum(ln_N, ln_N_cap)
        eps_arr = np.exp(ln_eps_clipped)
        N_arr = np.exp(ln_N_clipped)
        alpha   = alpha_0 * Mtil

        # SR rates
        g_sr_arr = np.array([gamma_tilde_sr(alpha, astar, n, l, m)
                             for (n, l, m) in LEVELS])

        # Annihilation rates
        g_ann_arr = np.array([gamma_tilde_ann(n, l, Mtil)
                              for (n, l, m) in LEVELS])

        # Transition contributions (NORM cancels: g_tr*N_j = g_tr*NORM*eps_j)
        dlneps_tr = np.zeros(N_LEVELS)
        for idx_i, idx_j, n_i, l_i, n_j, l_j in TR_PAIRS_IDX:
            g_tr = gamma_tilde_tr(n_i, l_i, n_j, l_j, Mtil)
            if g_tr > 0.0:
                # Use clipped N_arr values; ln_N_clipped ensures these are finite.
                dlneps_tr[idx_i] -= g_tr * N_arr[idx_j]   # = g_tr * NORM * eps_j
                dlneps_tr[idx_j] += g_tr * N_arr[idx_i]   # = g_tr * NORM * eps_i

        # d(ln_eps_i)/dtau  [identical form to d(lnN_i)/dtau]
        # Compute derivative in a numerically safe way; clip extremely large rates
        with np.errstate(over='ignore', invalid='ignore'):
            dlneps_arr = g_sr_arr - g_ann_arr * N_arr + dlneps_tr

        # Replace NaNs/Infs with safe values and cap derivative magnitude
        dlneps_arr = np.nan_to_num(dlneps_arr, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
        rate_cap = 1e200
        dlneps_arr = np.clip(dlneps_arr, -rate_cap, rate_cap)

        # BH mass: -(alpha_0/NORM)*dot(g_sr, NORM*eps) = -alpha_0*dot(g_sr, eps)
        dMtil = -alpha_0 * np.dot(g_sr_arr, eps_arr)

        # BH spin: dot(g_sr*NORM*eps,...)/NORM/Mtil^2 = dot(g_sr*eps,...)/Mtil^2
        with np.errstate(over='ignore', invalid='ignore'):
            dastar = (np.dot(g_sr_arr * eps_arr, 2.0 * alpha * astar - _M_ARR)
                      / Mtil**2)
        if not np.isfinite(dastar):
            dastar = 0.0
        # Ensure dMtil is finite
        if not np.isfinite(dMtil):
            dMtil = 0.0
        return list(dlneps_arr) + [dMtil, dastar]

    # ── Initial SR analysis ───────────────────────────────────────────
    g0_all    = {(n, l, m): gamma_tilde_sr(alpha_0, a_star_0, n, l, m)
                 for (n, l, m) in LEVELS}
    active_t0 = [(n, l, m) for (n, l, m) in LEVELS if g0_all[(n, l, m)] > 0]

    if not active_t0:
        print("\n  ✗  No levels are superradiant at the initial parameters.")
        for n, l, m in LEVELS[:7]:
            print(f"       |{n}{l}{m}⟩  m·Ω_H − α = {sr_margin(alpha_0, a_star_0, m):.4f}")
        print("\n  Increase ã₀ or decrease α₀ to enter the SR regime.")
        return None

    dom_level       = max(active_t0, key=lambda nlm: g0_all[nlm])
    dom_n, dom_l, dom_m = dom_level
    g0_dom          = g0_all[dom_level]

    # ── Time span ─────────────────────────────────────────────────────
    # We do NOT use terminal events.  The reason is fundamental: scipy's
    # adaptive-step RK45 evaluates event functions at intermediate
    # *rejected* trial steps.  Any event that uses mutable state (a
    # running-peak tracker) gets corrupted by these rejected evaluations,
    # causing the event to fire far too early — before the second level
    # has even started growing.
    #
    # Instead we pre-compute a tau_end that is large enough to cover:
    #   Phase 1 — fastest level (dom) grows from seed to saturation
    #              AND decays DECAY_DECADES below its peak
    #   Phase 2 — all higher-m levels that are still SR-active at the
    #              post-spindown spin (astar_f) grow and decay similarly
    #
    # This is the same two-phase structure as before, but:
    #   * decay_efolds are added to BOTH phases
    #   * Phase 2 uses the rate evaluated at astar_f (the actual spin
    #     when those levels start growing) not the initial spin

    NORM         = M0**2
    RATE_RATIO   = 1e40           # include phase-2 levels within 10^40 of fastest

    N_sat        = M0**2 * a_star_0
    n_efolds     = np.log(max(N_sat, 2.0))

    DECAY_DECADES = 10
    decay_efolds  = DECAY_DECADES * np.log(10.0)   # ≈ 23 e-folds

    astar_f = min(2.0 * alpha_0 * rhat_plus(a_star_0) / dom_m, a_star_0)

    # Phase 1: fastest level growth + decay
    tau_phase1 = (n_efolds + decay_efolds) / g0_dom

    # Phase 2: higher-m levels at the reduced spin
    g_floor    = g0_dom / RATE_RATIO
    tau_phase2 = 0.0
    phase2_levels = []
    for n, l, m in LEVELS:
        if m <= dom_m:
            continue
        g_af = gamma_tilde_sr(alpha_0, astar_f, n, l, m)
        if g_af >= g_floor:
            tau_i = (n_efolds + decay_efolds) / g_af
            if tau_i > tau_phase2:
                tau_phase2 = tau_i
            phase2_levels.append((n, l, m, g_af, tau_i))

    tau_end = tau_phase1 + tau_phase2
    if tau_end_factor is not None:
        tau_end *= tau_end_factor

    # ── Console output ────────────────────────────────────────────────
    print("=" * 65)
    print(f"BH Superradiance — {N_LEVELS} levels (m=l, n≤{max_n})  [hydrogenic]")
    print("=" * 65)
    print(f"  M_BH = {M_BH_solar} M_sun  =  {M0:.3e} M_Pl")
    print(f"  ã₀   = {a_star_0}")
    print(f"  α₀   = {alpha_0}   →  μ = {MU:.3e} M_Pl")
    print("\n  Levels tracked:")
    for n, l, m in LEVELS:
        g0_i   = g0_all[(n, l, m)]
        margin = sr_margin(alpha_0, a_star_0, m)
        if g0_i > 0:
            print(f"    |{n}{l}{m}⟩   Γ̃₀ = {g0_i:.3e}   τ_SR = {1/g0_i:.2e}   margin = {margin:.4f}")
        else:
            print(f"    |{n}{l}{m}⟩   Γ̃₀ = 0   (SR inactive, margin = {margin:.4f})")
    print("\n  Annihilation rates at initial parameters:")
    for n, l, m in LEVELS:
        if ann_available.get((n, l), False):
            g_ann_0 = gamma_tilde_ann(n, l, 1.0)
            if g_ann_0 > 0:
                print(f"    |{n}{l}{m}⟩   Γ̃_a = {g_ann_0:.3e}   τ_a = {1/g_ann_0:.2e}")
            else:
                print(f"    |{n}{l}{m}⟩   Γ̃_a = 0")
        else:
            print(f"    |{n}{l}{m}⟩   Γ̃_a = not computed")
    print(f"\n  Initially SR-active levels: {len(active_t0)}/{N_LEVELS}")
    print(f"  Dominant level: |{dom_n}{dom_l}{dom_m}⟩  Γ̃₀ = {g0_dom:.4e}")
    print(f"  e-folding τ_SR  = {1/g0_dom:.3e}  =  {1/g0_dom * TAU_TO_YR:.3e} yr")
    print(f"  ã_f ≈ {astar_f:.4f}  (spin after dominant level switches off)")
    if phase2_levels:
        print(f"  Phase-2 levels at ã_f (rate > {g_floor:.1e}):")
        for n, l, m, g_af, tau_i in sorted(phase2_levels, key=lambda x: -x[3]):
            print(f"    |{n}{l}{m}⟩   Γ̃(ã_f) = {g_af:.3e}"
                  f"   window = {tau_i:.2e} τ = {tau_i*TAU_TO_YR:.2e} yr")
    else:
        print(f"  No phase-2 levels above rate floor {g_floor:.1e}")
    print(f"  τ_end = {tau_end:.3e}  =  {tau_end * TAU_TO_YR:.3e} yr")
    print(f"  (no terminal events — simulation runs to τ_end)")

    # ── Initial conditions and integration ───────────────────────────
    ln_eps0 = np.log(N0) - 2.0 * np.log(M0)
    y0 = [ln_eps0] * N_LEVELS + [1.0, a_star_0]
    print("\nIntegrating ... ", end="", flush=True)
    sol = solve_ivp(
        odes,
        t_span       = (0.0, tau_end),
        y0           = y0,
        method       = "RK45",
        events       = [],           # no terminal events — see comment above
        rtol         = 1e-9,
        atol         = 1e-12,
        dense_output = False,
    )
    print("done.")

    tau   = sol.t
    Mtil  = sol.y[N_LEVELS]
    astar = sol.y[N_LEVELS + 1]

    # ── Derived quantities ────────────────────────────────────────────
    # The solver evolved ln_eps; recover eps and N.
    ln_eps_raw = sol.y[:N_LEVELS]
    eps_all    = np.exp(np.clip(ln_eps_raw, -500.0, 700.0))   # (N_LEVELS, n_time)
    N_all      = NORM * eps_all                                # physical occupation
    lnN_all    = np.log(np.maximum(N_all, 1e-300))            # for downstream compatibility

    alpha   = alpha_0 * Mtil
    M_BH    = Mtil * M0

    J_cloud = _M_ARR @ N_all
    M_cloud = MU * N_all.sum(axis=0)
    J_BH    = astar * Mtil**2 * M0**2
    J_total = J_BH + J_cloud
    dJ_frac = np.abs((J_total - J_total[0]) / J_total[0])
    t_yr    = tau * TAU_TO_YR

    M_irr  = np.array([irreducible_mass(m, a) for m, a in zip(M_BH, astar)])
    E_rot  = M_BH - M_irr
    dM_irr = M_irr - M_irr[0]

    # SR rates over time
    g_sr_all = np.array([
        [gamma_tilde_sr(a, s, n, l, m) for a, s in zip(alpha, astar)]
        for (n, l, m) in LEVELS
    ])

    # Annihilation rates over time
    g_ann_all = np.zeros((N_LEVELS, len(tau)))
    for k, (n, l, m) in enumerate(LEVELS):
        if ann_available.get((n, l), False):
            g_ann_all[k] = np.array([gamma_tilde_ann(n, l, Mt) for Mt in Mtil])

    # ── GW strain from annihilation ──────────────────────────────────
    omega_ann_all = np.array([
        2.0 * MU * (1.0 - alpha**2 / (2.0 * n**2))
        for (n, l, m) in LEVELS
    ])
    Gamma_a_phys = g_ann_all * MU
    with np.errstate(invalid='ignore', divide='ignore'):
        h_all = N_all * np.sqrt(
            np.maximum(8.0 * Gamma_a_phys / (KPC_PLANCK**2 * omega_ann_all), 0.0)
        )
    h_all = np.nan_to_num(h_all, nan=0.0, posinf=0.0)

    # Dimensionless strain amplitude:  h_dimless = h * r_1kpc / M0
    # Physical strain at any mass M and distance r:  h = h_dimless * M_Pl / r_Pl
    h_dimless_all = h_all * KPC_PLANCK / M0   # (N_LEVELS, n_time)
    f_gw_hz = np.array([
        2.0 * MU * (1.0 - alpha_0**2 / (2.0 * n**2)) * OMEGA_PL_TO_HZ
        for (n, l, m) in LEVELS
    ])
    P_GW_total = (2.0 * g_ann_all * N_all**2).sum(axis=0)

    # ── GW strain from transitions ────────────────────────────────────
    n_tr_pairs = len(TR_PAIRS_IDX)
    g_tr_matrix = np.zeros((n_tr_pairs, len(tau)))
    h_tr_matrix = np.zeros((n_tr_pairs, len(tau)))
    f_tr_hz = np.zeros(n_tr_pairs)

    for p, (idx_i, idx_j, n_i, l_i, n_j, l_j) in enumerate(TR_PAIRS_IDX):
        g_tr_matrix[p] = np.array([gamma_tilde_tr(n_i, l_i, n_j, l_j, Mt) for Mt in Mtil])
        omega_tr_all = MU * (alpha**2 / 2.0) * (1.0 / n_j**2 - 1.0 / n_i**2)
        omega_tr_all = np.maximum(omega_tr_all, 1e-300)
        Gamma_tr_phys = g_tr_matrix[p] * MU
        with np.errstate(invalid='ignore', divide='ignore'):
            h_pair = (np.sqrt(np.maximum(N_all[idx_i] * N_all[idx_j], 0.0))
                      * np.sqrt(np.maximum(
                          8.0 * Gamma_tr_phys / (KPC_PLANCK**2 * omega_tr_all), 0.0)))
        h_tr_matrix[p] = np.nan_to_num(h_pair, nan=0.0, posinf=0.0)
        omega_tr_0 = MU * (alpha_0**2 / 2.0) * (1.0 / n_j**2 - 1.0 / n_i**2)
        f_tr_hz[p] = omega_tr_0 * OMEGA_PL_TO_HZ

    h_dimless_tr = h_tr_matrix * KPC_PLANCK / M0   # (n_tr_pairs, n_time)

    # ── Summary prints ───────────────────────────────────────────────
    n_ann_active = sum(1 for (n,l) in ann_available if ann_available[(n,l)])
    n_tr_active = len(TR_PAIRS_IDX)
    print(f"\nResults")
    print(f"  End:  τ = {tau[-1]:.4e}   t = {t_yr[-1]:.4f} yr")
    print(f"  ã   : {a_star_0:.4f} → {astar[-1]:.4f}  (Δã = {a_star_0 - astar[-1]:.4f})")
    print(f"  M   : {M_BH_solar:.3f} → {Mtil[-1]*M_BH_solar:.5f} M_sun"
          f"  (ΔM/M₀ = {1 - Mtil[-1]:.5f})")
    print(f"\n  Peak occupation numbers:")
    for k, (n, l, m) in enumerate(LEVELS):
        peak = lnN_all[k].max() / np.log(10)
        if peak > 0.5:
            ann_tag = ("[ann. active]" if ann_available.get((n, l), False) else "[ann. not computed]")
            print(f"    |{n}{l}{m}⟩   log₁₀N_max = {peak:.2f}   {ann_tag}")
    M_irr0_sol = M_irr[0] / M_SUN_PLANCK
    M_irrf_sol = M_irr[-1] / M_SUN_PLANCK
    E_rot0_sol = E_rot[0] / M_SUN_PLANCK
    E_rotf_sol = E_rot[-1] / M_SUN_PLANCK
    print(f"\n  Christodoulou-Ruffini decomposition:")
    print(f"  M_irr : {M_irr0_sol:.5e} → {M_irrf_sol:.5e} M_sun"
          f"  (ΔM_irr = +{M_irrf_sol - M_irr0_sol:.5e}, area theorem ✓)")
    print(f"  E_rot : {E_rot0_sol:.5e} → {E_rotf_sol:.5e} M_sun")
    print(f"  M_cloud total : {M_cloud[-1]/M_SUN_PLANCK:.5e} M_sun")
    print(f"  Annihilation  : {n_ann_active}/{N_LEVELS} levels have computed rates")
    print(f"  Transitions   : {n_tr_active} pairs have computed rates")

    # ── Pack results ──────────────────────────────────────────────────
    res = SimulationResults(
        M_BH_solar=M_BH_solar,
        a_star_0=a_star_0,
        alpha_0=alpha_0,
        max_n=max_n,
        M0=M0,
        MU=MU,
        TAU_TO_YR=TAU_TO_YR,
        LEVELS=LEVELS,
        _L_ARR=_L_ARR,
        _M_ARR=_M_ARR,
        N_LEVELS=N_LEVELS,
        ann_available=ann_available,
        tr_pairs_idx=TR_PAIRS_IDX,
        NORM=NORM,
        tau=tau,
        lnN_all=lnN_all,
        eps_all=eps_all,
        Mtil=Mtil,
        astar=astar,
        t_yr=t_yr,
        N_all=N_all,
        alpha=alpha,
        M_BH=M_BH,
        J_cloud=J_cloud,
        M_cloud=M_cloud,
        J_BH=J_BH,
        J_total=J_total,
        dJ_frac=dJ_frac,
        M_irr=M_irr,
        E_rot=E_rot,
        dM_irr=dM_irr,
        g_sr_all=g_sr_all,
        g_ann_all=g_ann_all,
        h_all=h_all,
        h_dimless_all=h_dimless_all,
        f_gw_hz=f_gw_hz,
        P_GW_total=P_GW_total,
        g_tr_matrix=g_tr_matrix,
        h_tr_matrix=h_tr_matrix,
        h_dimless_tr=h_dimless_tr,
        f_tr_hz=f_tr_hz,
        astar_f=astar_f,
        t_final_yr=t_yr[-1],
        active_levels_t0=len(active_t0),
        dominant_level=dom_level,
    )
    return res


# ══════════════════════════════════════════════════════════════════════
# Dimensionless file I/O and mass rescaling
# ══════════════════════════════════════════════════════════════════════

def save_dimensionless(results: SimulationResults, filepath: str = None):
    """
    Save the universal dimensionless solution to a compressed NPZ file.

    The arrays eps_all, h_dimless_all, h_dimless_tr are functions of
    (alpha_0, a_star_0) only — they do not change with BH mass.
    Use rescale_to_mass() to recover physical quantities at any mass M.

    Rescaling rules
    ---------------
    M_Pl = M_solar * 9.137e37               [M_Planck]
    t    = tau * (M_solar * 4.927e-6) / (alpha_0 * 3.156e7)    [yr]
    N_i  = M_Pl^2 * eps_i
    h(r) = h_dimless * M_Pl / r_Pl          (r_Pl = r_kpc * 1.910e54)
    f(M) = f_gw_hz * (M0_solar / M_solar)   [Hz, at fixed alpha_0]
    """
    if results is None:
        print("No results to save.")
        return

    if filepath is None:
        data_dir = os.path.join(_THIS_DIR, 'Data')
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir,
            f"dimless_alpha{results.alpha_0:.3f}_a{results.a_star_0:.2f}.npz")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    level_labels = np.array([f"{n}{l}{m}" for (n, l, m) in results.LEVELS])
    tr_labels = np.array(
        [f"{n_i}{l_i}{l_i}->{n_j}{l_j}{l_j}"
         for (_, _, n_i, l_i, n_j, l_j) in results.tr_pairs_idx]
        if results.tr_pairs_idx else []
    )

    np.savez_compressed(
        filepath,
        alpha_0      = results.alpha_0,
        a_star_0     = results.a_star_0,
        M0_solar     = results.M_BH_solar,
        M0_planck    = results.M0,
        NORM         = results.NORM,
        tau          = results.tau,
        eps_all      = results.eps_all,
        Mtil         = results.Mtil,
        astar        = results.astar,
        alpha        = results.alpha,
        g_sr_all     = results.g_sr_all,
        g_ann_all    = results.g_ann_all,
        g_tr_matrix  = results.g_tr_matrix,
        h_dimless_all = results.h_dimless_all,
        h_dimless_tr = results.h_dimless_tr,
        f_gw_hz      = results.f_gw_hz,
        f_tr_hz      = results.f_tr_hz,
        level_labels = level_labels,
        tr_labels    = tr_labels,
    )
    print(f"Dimensionless results saved to {filepath}")

    guide = filepath.replace('.npz', '_rescaling.txt')
    with open(guide, 'w', encoding='utf-8') as f:
        f.write("Superradiance simulation — dimensionless output\n")
        f.write("=" * 60 + "\n")
        f.write(f"alpha_0  = {results.alpha_0}\n")
        f.write(f"a_star_0 = {results.a_star_0}\n")
        f.write(f"M0_solar = {results.M_BH_solar} M_sun\n")
        f.write(f"M0_Pl    = {results.M0:.6e} M_Pl\n\n")
        f.write("Arrays tau, eps_all, Mtil, astar, h_dimless_all, h_dimless_tr\n")
        f.write("are universal at fixed (alpha_0, a_star_0).\n\n")
        f.write("Rescaling to mass M [solar masses]\n")
        f.write("-" * 40 + "\n")
        f.write(f"  M_Pl = M_solar * {M_SUN_PLANCK:.6e}\n")
        f.write(f"  t [yr] = tau * M_solar * {GM_SUN_OVER_C3:.6e} / (alpha_0 * {YEAR_S:.6e})\n")
        f.write(f"  N_i    = M_Pl^2 * eps_i\n")
        f.write(f"  h(r)   = h_dimless * M_Pl / (r_kpc * {KPC_PLANCK:.6e})\n")
        f.write(f"  f(M)   = f_gw_hz * ({results.M_BH_solar} / M_solar)  [Hz]\n")
        f.write(f"\n  h ∝ M  at fixed alpha_0 and fixed distance\n")
        f.write(f"  t ∝ M  at fixed alpha_0\n")
        f.write(f"  f ∝ 1/M at fixed alpha_0\n")
    print(f"Rescaling guide saved to {guide}")


def rescale_to_mass(filepath: str, M_new_solar: float, r_kpc: float = 1.0):
    """
    Load a dimensionless NPZ file and return physical quantities at a new mass.

    Parameters
    ----------
    filepath     : path to .npz written by save_dimensionless()
    M_new_solar  : target BH mass [solar masses]
    r_kpc        : source distance [kpc]

    Returns
    -------
    dict with physical quantities rescaled to M_new_solar at r_kpc.
    """
    data         = np.load(filepath, allow_pickle=True)
    alpha_0      = float(data['alpha_0'])
    M0_solar     = float(data['M0_solar'])
    M_new_planck = M_new_solar * M_SUN_PLANCK
    r_planck     = r_kpc * KPC_PLANCK
    TAU_TO_YR_new = (M_new_solar * GM_SUN_OVER_C3) / (alpha_0 * YEAR_S)

    return {
        # Dimensionless (unchanged by rescaling)
        'tau':        data['tau'],
        'eps_all':    data['eps_all'],
        'Mtil':       data['Mtil'],
        'astar':      data['astar'],
        'alpha':      data['alpha'],
        # Physical time axis
        't_yr':       data['tau'] * TAU_TO_YR_new,
        # Physical occupation
        'N_all':      M_new_planck**2 * data['eps_all'],
        # Physical GW strain at r_kpc
        'h_ann':      data['h_dimless_all'] * M_new_planck / r_planck,
        'h_tr':       data['h_dimless_tr']  * M_new_planck / r_planck,
        # GW frequencies rescaled to new mass (f ∝ 1/M at fixed alpha_0)
        'f_gw_hz':    data['f_gw_hz'] * (M0_solar / M_new_solar),
        'f_tr_hz':    data['f_tr_hz'] * (M0_solar / M_new_solar),
        # Metadata
        'M_planck':   M_new_planck,
        'alpha_0':    alpha_0,
        'a_star_0':   float(data['a_star_0']),
        'r_kpc':      r_kpc,
        'level_labels': data['level_labels'],
        'tr_labels':    data['tr_labels'],
    }


# ══════════════════════════════════════════════════════════════════════
# Peak characteristic analysis
# ══════════════════════════════════════════════════════════════════════

def compute_peak_characteristics(results: SimulationResults):
    """
    Compute for each annihilation level and transition pair:
    - peak strain h_max (at 1 kpc)
    - FWHM (full width at half maximum) in years
    - time of peak t_peak [yr]
    - frequency of emission [Hz]

    Returns two lists of dicts, sorted by decreasing peak strain.
    """
    if results is None:
        return [], []

    t_yr = results.t_yr
    LEVELS = results.LEVELS
    h_all = results.h_all
    f_gw_hz = results.f_gw_hz

    # ── Annihilation peaks ────────────────────────────────────────────
    ann_char = []
    for k, (n, l, m) in enumerate(LEVELS):
        h = h_all[k]
        h_max = np.max(h)
        if h_max <= 0:
            continue

        idx_peak = int(np.argmax(h))
        t_peak = t_yr[idx_peak]
        half_max = h_max / 2.0

        # Find indices where strain >= half max
        above = np.where(h >= half_max)[0]
        fwhm = 0.0
        if len(above) > 0:
            # Split into contiguous blocks
            blocks = np.split(above, np.where(np.diff(above) != 1)[0] + 1)
            # Find block containing the peak index
            for blk in blocks:
                if idx_peak in blk:
                    if len(blk) > 1:
                        fwhm = t_yr[blk[-1]] - t_yr[blk[0]]
                    break

        ann_char.append({
            'label': f"|{n}{l}{m}⟩",
            'peak_strain': h_max,
            'fwhm_yr': fwhm,
            't_peak_yr': t_peak,
            'frequency_hz': f_gw_hz[k]
        })

    # ── Transition peaks ──────────────────────────────────────────────
    tr_char = []
    TR_PAIRS_IDX = results.tr_pairs_idx
    h_tr_matrix = results.h_tr_matrix
    f_tr_hz = results.f_tr_hz

    for p, (idx_i, idx_j, n_i, l_i, n_j, l_j) in enumerate(TR_PAIRS_IDX):
        h = h_tr_matrix[p]
        h_max = np.max(h)
        if h_max <= 0:
            continue

        idx_peak = int(np.argmax(h))
        t_peak = t_yr[idx_peak]
        half_max = h_max / 2.0

        above = np.where(h >= half_max)[0]
        fwhm = 0.0
        if len(above) > 0:
            blocks = np.split(above, np.where(np.diff(above) != 1)[0] + 1)
            for blk in blocks:
                if idx_peak in blk:
                    if len(blk) > 1:
                        fwhm = t_yr[blk[-1]] - t_yr[blk[0]]
                    break

        tr_char.append({
            'label': f"|{n_i}{l_i}{l_i}⟩→|{n_j}{l_j}{l_j}⟩",
            'peak_strain': h_max,
            'fwhm_yr': fwhm,
            't_peak_yr': t_peak,
            'frequency_hz': f_tr_hz[p]
        })

    # Sort by decreasing peak strain
    ann_char.sort(key=lambda x: x['peak_strain'], reverse=True)
    tr_char.sort(key=lambda x: x['peak_strain'], reverse=True)

    return ann_char, tr_char


def print_peak_tables(ann_char, tr_char):
    """Print formatted tables of peak characteristics to console."""
    # ── Annihilation table ────────────────────────────────────────────
    if ann_char:
        print("\n" + "=" * 110)
        print("  Annihilation Peak Characteristics (sorted by decreasing peak strain at 1 kpc)")
        print("=" * 110)
        header = (f"{'Level':<16} {'Peak Strain':>15} {'FWHM [yr]':>15} "
                  f"{'t_peak [yr]':>18} {'Frequency [Hz]':>18}")
        print(header)
        print("-" * 110)
        for entry in ann_char:
            print(f"{entry['label']:<16} {entry['peak_strain']:>15.3e} "
                  f"{entry['fwhm_yr']:>15.3e} {entry['t_peak_yr']:>18.3e} "
                  f"{entry['frequency_hz']:>18.3e}")
    else:
        print("\nNo annihilation peaks to display.")

    # ── Transition table ──────────────────────────────────────────────
    if tr_char:
        print("\n" + "=" * 110)
        print("  Transition Peak Characteristics (sorted by decreasing peak strain at 1 kpc)")
        print("=" * 110)
        header = (f"{'Transition':<28} {'Peak Strain':>15} {'FWHM [yr]':>15} "
                  f"{'t_peak [yr]':>18} {'Frequency [Hz]':>18}")
        print(header)
        print("-" * 110)
        for entry in tr_char:
            print(f"{entry['label']:<28} {entry['peak_strain']:>15.3e} "
                  f"{entry['fwhm_yr']:>15.3e} {entry['t_peak_yr']:>18.3e} "
                  f"{entry['frequency_hz']:>18.3e}")
    else:
        print("\nNo transition peaks to display.")

    print("=" * 110)


def save_peak_tables(ann_char, tr_char, results: SimulationResults, output_dir="Data"):
    """
    Save peak characteristic tables as .dat files in output_dir.
    File names contain process type, BH mass (scientific), alpha, and spin.
    """
    if results is None:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Format mass: e.g. 1e-11 → "1e-11"
    mass_str = f"M{results.M_BH_solar:.0e}".replace('e-0', 'e-').replace('e+0', 'e')
    # Format alpha and spin: e.g., alpha0.60, a0.65
    alpha_str = f"alpha{results.alpha_0:.3f}".rstrip('0').rstrip('.') if '.' in f"{results.alpha_0:.2f}" else f"alpha{results.alpha_0:.2f}"
    spin_str = f"a{results.a_star_0:.2f}".rstrip('0').rstrip('.') if '.' in f"{results.a_star_0:.2f}" else f"a{results.a_star_0:.2f}"

    fieldnames = ['label', 'peak_strain', 'fwhm_yr', 't_peak_yr', 'frequency_hz']

    # Annihilation table
    if ann_char:
        ann_filename = f"peaks_annihilation_{mass_str}_{alpha_str}_{spin_str}.dat"
        ann_path = os.path.join(output_dir, ann_filename)
        with open(ann_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            for entry in ann_char:
                writer.writerow({k: entry[k] for k in fieldnames})
        print(f"Saved annihilation peak table to {ann_path}")

    # Transition table
    if tr_char:
        tr_filename = f"peaks_transitions_{mass_str}_{alpha_str}_{spin_str}.dat"
        tr_path = os.path.join(output_dir, tr_filename)
        with open(tr_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            for entry in tr_char:
                writer.writerow({k: entry[k] for k in fieldnames})
        print(f"Saved transition peak table to {tr_path}")

# ══════════════════════════════════════════════════════════════════════
# Plotting functions
# ══════════════════════════════════════════════════════════════════════

# Colour scheme: one colour per l value
_L_COLORS = {
    1: "#1f77b4",   # blue
    2: "#d62728",   # red
    3: "#2ca02c",   # green
    4: "#9467bd",   # purple
    5: "#8c564b",   # brown
    6: "#e377c2",   # pink
    7: "#7f7f7f",   # grey
}
_LSTYLES = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,2)), (0,(1,1))]

def level_style(n, l):
    """Return (color, linestyle, linewidth) for level (n,l)."""
    base_col = _L_COLORS.get(l, 'k')
    n_idx = n - (l + 1)          # 0 for lowest n in this l-group
    ls = _LSTYLES[n_idx % len(_LSTYLES)]
    lw = max(2.0 - 0.2 * n_idx, 0.8)
    return base_col, ls, lw


def plot_occupations_and_spin(results: SimulationResults, save_path=None):
    """Main plot: spin panel (top) + log10 N(t) for all levels (bottom).

    Changes vs. original:
      - ã_f annotation moved to the right edge of the spin panel.
      - No legend: inline coloured text labels stacked vertically at the
        centre-height / left of the occupation panel.
      - Colour scheme: n=2 levels → shades of blue,
                       n=3 levels → dark shades of red.
    """
    if results is None:
        print("No simulation results to plot.")
        return

    t_yr      = results.t_yr
    astar     = results.astar
    log10N_all = results.lnN_all / np.log(10)
    LEVELS    = results.LEVELS
<<<<<<< HEAD
=======

    # ── Colour palettes ───────────────────────────────────────────────
    _N2_BLUES = ["#1565C0", "#42A5F5", "#0288D1", "#29B6F6"]   # darkest first
    _N3_REDS  = ["#8B0000", "#C62828", "#D32F2F", "#E57373"]   # darkest first
    _LS_CYCLE = ["-", "--", "-.", ":"]

    # Pre-compute (colour, linestyle, linewidth) for every level in order
    _n2_idx, _n3_idx = 0, 0
    level_style_map = {}
    for (n, l, m) in LEVELS:
        if n == 2:
            col = _N2_BLUES[_n2_idx % len(_N2_BLUES)]
            _n2_idx += 1
        elif n == 3:
            col = _N3_REDS[_n3_idx % len(_N3_REDS)]
            _n3_idx += 1
        else:
            col, _, _ = level_style(n, l)        # fallback for n > 3
        ls = _LS_CYCLE[(l - 1) % len(_LS_CYCLE)]
        level_style_map[(n, l, m)] = (col, ls, 1.8)
>>>>>>> 3b9e21da0657b2c765a7a216e3ca9d6366e9592c

    # ── Colour palettes ───────────────────────────────────────────────
    _N2_BLUES = ["#1565C0", "#42A5F5", "#0288D1", "#29B6F6"]   # darkest first
    _N3_REDS  = ["#8B0000", "#C62828", "#D32F2F", "#E57373"]   # darkest first
    _LS_CYCLE = ["-", "--", "-.", ":"]

<<<<<<< HEAD
    # Pre-compute (colour, linestyle, linewidth) for every level in order
    _n2_idx, _n3_idx = 0, 0
    level_style_map = {}
    for (n, l, m) in LEVELS:
        if n == 2:
            col = _N2_BLUES[_n2_idx % len(_N2_BLUES)]
            _n2_idx += 1
        elif n == 3:
            col = _N3_REDS[_n3_idx % len(_N3_REDS)]
            _n3_idx += 1
        else:
            col, _, _ = level_style(n, l)        # fallback for n > 3
        ls = _LS_CYCLE[(l - 1) % len(_LS_CYCLE)]
        level_style_map[(n, l, m)] = (col, ls, 1.8)

    color_spin = "k"

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

    fig_main = plt.figure(figsize=(5, 4))
=======
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

    fig_main = plt.figure(figsize=(8, 6))
>>>>>>> 3b9e21da0657b2c765a7a216e3ca9d6366e9592c
    gs_main  = GridSpec(2, 1, figure=fig_main,
                        height_ratios=[0.5, 3.5], hspace=0.1)
    ax_spin  = fig_main.add_subplot(gs_main[0])
    ax_main  = fig_main.add_subplot(gs_main[1], sharex=ax_spin)

    # ── Spin panel ────────────────────────────────────────────────────
    ax_spin.plot(t_yr, astar, color='k', lw=1.2)
    ax_spin.set_ylabel(r"$\tilde{a}$", fontsize=11)
    ax_spin.tick_params(axis="y", labelsize=9)
    ax_spin.tick_params(axis="x", labelbottom=False)
    ax_spin.set_ylim(-0.05, 1.10)
    ax_spin.set_yticks([0.0, 0.5, 1.0])
    ax_spin.grid(True, alpha=0.25, linestyle="--")
    ax_spin.axhline(astar[-1], ls=":", color=color_spin, lw=0.8, alpha=0.55)

    # ã_f label pinned to the RIGHT edge of the axes at the final spin value
    ax_spin.annotate(
        fr"$\tilde{{a}}_f = {astar[-1]:.3f}$",
        xy=(1.0, astar[-1]),
        xycoords=("axes fraction", "data"),
        xytext=(-6, 5), textcoords="offset points",
        fontsize=8, color=color_spin,
        ha="right", va="bottom",
    )
    # ã_0 label at the left / top of the spin curve
    ax_spin.annotate(
        fr"$\tilde{{a}}_0 = {results.a_star_0}$",
        xy=(t_yr[t_yr > 0][0] if np.any(t_yr > 0) else t_yr[1], results.a_star_0),
        xytext=(6, -10), textcoords="offset points",
        fontsize=8, color=color_spin,
    )
    ax_spin.set_xscale("log")

    # ── Occupation panel ──────────────────────────────────────────────
    active_labels = []   # (latex_str, colour, peak_log10N) for inline labels

    for k, (n, l, m) in enumerate(LEVELS):
        col, ls, lw = level_style_map[(n, l, m)]
        is_active = log10N_all[k].max() > 0.5
        ax_main.plot(
            t_yr, log10N_all[k],
            color=col, ls=ls, lw=lw,
            alpha=1.0 if is_active else 0.25,
        )
        if is_active:
            active_labels.append(
                (rf"$|{n}{l}{m}\rangle$", col, float(log10N_all[k].max()))
            )

    ax_main.set_xlabel(r"$t$  [yr]", fontsize=12)
    ax_main.set_ylabel(r"$\log_{10}\, N$", fontsize=12)
    ax_main.grid(True, alpha=0.25, linestyle="--")
    ax_main.set_ylim(0, log10N_all.max() * 1.1)
    t_min_pos = t_yr[t_yr > 0][0] if np.any(t_yr > 0) else t_yr[1]
    ax_main.set_xlim(t_min_pos, t_yr[-1])
    ax_main.set_xscale("log")

    # ── Inline coloured text labels ───────────────────────────────────
    # Sort by descending peak so the tallest curve appears at the top
    active_labels.sort(key=lambda x: x[2], reverse=True)
    n_active = len(active_labels)

    if n_active > 0:
        x_axes  = 0.05          # left side in axes-fraction coords
        y_gap   = 0.08          # vertical spacing between labels (axes fraction)
        total_h = y_gap * (n_active - 1)
        y_top   = 0.50 + total_h / 2.0   # centre the block around y=0.5

        for i, (lbl, col, _) in enumerate(active_labels):
            ax_main.text(
                x_axes,
                y_top - i * y_gap,
                lbl,
                transform=ax_main.transAxes,
                fontsize=11, color=col,
                ha="left", va="center",
            )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig_main.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Main figure saved to {save_path}")
    return fig_main


def plot_gw_transitions(results: SimulationResults, save_path=None):
    """GW strain from transitions: spin + h_tr(t) per pair."""
    if results is None:
        return
    t_yr = results.t_yr
    astar = results.astar
    h_tr_matrix = results.h_tr_matrix
    f_tr_hz = results.f_tr_hz
    TR_PAIRS_IDX = results.tr_pairs_idx
    if len(TR_PAIRS_IDX) == 0 or not np.any(h_tr_matrix > 0):
        print("No transition GW strain to plot.")
        return

    color_spin = "firebrick"
    fig_tr = plt.figure(figsize=(8, 6))
    gs_tr = GridSpec(2, 1, figure=fig_tr, height_ratios=[0.5, 3.5], hspace=0.1)
    ax_tr_spin = fig_tr.add_subplot(gs_tr[0])
    ax_tr_h = fig_tr.add_subplot(gs_tr[1], sharex=ax_tr_spin)

    ax_tr_spin.plot(t_yr, astar, color='k', lw=1.2)
    ax_tr_spin.set_ylabel(r"$\tilde{a}$", fontsize=11)
    ax_tr_spin.tick_params(axis="y", labelsize=9)
    ax_tr_spin.tick_params(axis="x", labelbottom=False)
    ax_tr_spin.set_ylim(-0.05, 1.10)
    ax_tr_spin.set_yticks([0.0, 0.5, 1.0])
    ax_tr_spin.grid(True, alpha=0.25, linestyle="--")
    ax_tr_spin.axhline(astar[-1], ls=":", color=color_spin, lw=0.8, alpha=0.55)
    ax_tr_spin.set_xscale('log')

    for p, (idx_i, idx_j, n_i, l_i, n_j, l_j) in enumerate(TR_PAIRS_IDX):
        if h_tr_matrix[p].max() <= 0:
            continue
        col, ls, lw = level_style(n_i, l_i)
        f_hz = f_tr_hz[p]
        if f_hz >= 1e9: f_str = fr"$f={f_hz/1e9:.2g}$\,GHz"
        elif f_hz >= 1e6: f_str = fr"$f={f_hz/1e6:.2g}$\,MHz"
        elif f_hz >= 1e3: f_str = fr"$f={f_hz/1e3:.2g}$\,kHz"
        else: f_str = fr"$f={f_hz:.2g}$\,Hz"
        label = rf"$|{n_i}{l_i}{l_i}\rangle \to |{n_j}{l_j}{l_j}\rangle$\;({f_str})"
        ax_tr_h.semilogy(t_yr, np.maximum(h_tr_matrix[p], 1e-100),
                         color=col, ls=ls, lw=lw, label=label)
    ax_tr_h.set_xlabel(r"$t$  [yr]", fontsize=12)
    ax_tr_h.set_ylabel(r"$h_{\rm tr}$ (at 1\,kpc)", fontsize=12)
    ax_tr_h.grid(True, alpha=0.25, which="both", linestyle="--")
    ax_tr_h.legend(fontsize=8, loc="upper left", ncol=1)
    ax_tr_h.set_xscale('log')
    t_min_pos = t_yr[t_yr > 0][0] if np.any(t_yr > 0) else t_yr[1]
    ax_tr_h.set_xlim(t_min_pos, t_yr[-1])

    fig_tr.suptitle(
        fr"GW strain from axion transitions — $m=l$ levels ($n\leq{results.max_n}$), "
        fr"$M_{{\rm BH}} = {results.M_BH_solar}\,M_\odot$, "
        fr"$\alpha_0 = {results.alpha_0}$, distance $= 1\,\rm kpc$",
        fontsize=9, y=1.01)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig_tr.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Transition GW strain figure saved to {save_path}")
    return fig_tr


def plot_annihilation_rates(results: SimulationResults, save_path=None):
    """Diagnostic plot: annihilation rates over time."""
    if results is None:
        return
    g_ann_all = results.g_ann_all
    if not np.any(g_ann_all > 0):
        print("No positive annihilation rates to plot.")
        return
    t_yr = results.t_yr
    LEVELS = results.LEVELS

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for k, (n, l, m) in enumerate(LEVELS):
        if g_ann_all[k].max() > 0:
            col, ls, lw = level_style(n, l)
            ax3.semilogy(t_yr, g_ann_all[k], color=col, ls=ls, lw=lw,
                         label=rf"$|{n}{l}{m}\rangle$")
    ax3.set_xlabel(r"$t$  [yr]", fontsize=12)
    ax3.set_ylabel(r"$\tilde{\Gamma}_a$", fontsize=12)
    ax3.set_title("Annihilation rates over time")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25, which="both", linestyle="--")
    ax3.set_xscale('log')
    t_min_pos = t_yr[t_yr > 0][0] if np.any(t_yr > 0) else t_yr[1]
    ax3.set_xlim(t_min_pos, t_yr[-1])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig3.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Annihilation rates plot saved to {save_path}")
    return fig3


def plot_diagnostics(results: SimulationResults, save_path=None):
    """2x2 diagnostic plot: SR rates, energy decomposition, angular momentum, phase-space."""
    if results is None:
        return
    t_yr = results.t_yr
    LEVELS = results.LEVELS
    g_sr_all = results.g_sr_all
    M_irr = results.M_irr
    E_rot = results.E_rot
    M_cloud = results.M_cloud
    J_BH = results.J_BH
    J_cloud = results.J_cloud
    J_total = results.J_total
    alpha = results.alpha
    astar = results.astar
    _M_ARR = results._M_ARR
    M_SUN_PLANCK = 9.137e37  # reuse local constant

    t_min_pos = t_yr[t_yr > 0][0] if np.any(t_yr > 0) else t_yr[1]
    t_max = t_yr[-1]

    fig2 = plt.figure(figsize=(13, 9))
    gs2 = GridSpec(2, 2, figure=fig2, hspace=0.42, wspace=0.34)
    s1 = fig2.add_subplot(gs2[0, 0])
    s2 = fig2.add_subplot(gs2[0, 1])
    s3 = fig2.add_subplot(gs2[1, 0])
    s4 = fig2.add_subplot(gs2[1, 1])
    tl = r"$t$  [yr]"

    # S1: SR rates
    for k, (n, l, m) in enumerate(LEVELS):
        col, ls, lw = level_style(n, l)
        label = rf"$|{n}{l}{m}\rangle$"
        show_label = g_sr_all[k].max() > g_sr_all[0].max() * 1e-6
        s1.semilogy(t_yr, np.maximum(g_sr_all[k], 1e-100),
                    color=col, ls=ls, lw=lw,
                    label=label if show_label else "_nolegend_")
    s1.set_xlabel(tl)
    s1.set_ylabel(r"$\tilde{\Gamma}_{\rm SR}$")
    s1.set_title("Superradiance rates")
    s1.legend(fontsize=7, ncol=2)
    s1.set_xscale('log')
    s1.set_xlim(t_min_pos, t_max)
    s1.grid(True, alpha=0.25, which="both", linestyle="--")

    # S2: Christodoulou-Ruffini decomposition
    s2.plot(t_yr, M_irr / M_SUN_PLANCK, color="firebrick", lw=1.8,
            label=r"$M_{\rm irr}$")
    s2.plot(t_yr, E_rot / M_SUN_PLANCK, color="darkorange", lw=1.8,
            label=r"$E_{\rm rot}$")
    s2.plot(t_yr, M_cloud / M_SUN_PLANCK, color="royalblue", lw=1.8, ls="--",
            label=r"$M_{\rm cloud}$")
    s2.axhline(E_rot[0] / M_SUN_PLANCK, ls=":", color="darkorange", lw=0.9, alpha=0.5)
    s2.set_xlabel(tl); s2.set_ylabel(r"Mass-energy  $[M_\odot]$")
    s2.set_title("Christodoulou--Ruffini decomposition")
    s2.legend(fontsize=9)
    s2.set_xscale('log')
    s2.set_xlim(t_min_pos, t_max)
    s2.grid(True, alpha=0.25, linestyle="--")

    # S3: Angular momentum budget
    J0 = J_total[0]
    s3.plot(t_yr, J_BH / J0, color="firebrick", lw=1.8,
            label=r"$J_{\rm BH}/J_0$")
    s3.plot(t_yr, J_cloud / J0, color="royalblue", lw=1.8,
            label=r"$J_{\rm cloud}/J_0$")
    s3.plot(t_yr, J_total / J0, "k--", lw=1.2, alpha=0.55,
            label=r"$J_{\rm total}$ (conserved)")
    s3.set_xlabel(tl); s3.set_ylabel(r"$J / J_0$")
    s3.set_title("Angular momentum budget")
    s3.legend(fontsize=9)
    s3.set_xscale('log')
    s3.set_xlim(t_min_pos, t_max)
    s3.grid(True, alpha=0.25, linestyle="--")

    # S4: Phase-space trajectory (α, ã)
    a_line = np.linspace(0.01, 0.9999, 500)
    for m_val in sorted(set(_M_ARR)):
        alpha_bnd = m_val * a_line / (2.0 * (1.0 + np.sqrt(1.0 - a_line**2)))
        col = _L_COLORS.get(m_val, 'k')
        s4.plot(alpha_bnd, a_line, ls="--", lw=1.0, color=col, alpha=0.7,
                label=rf"$m={m_val}$ boundary")
    s4.plot(alpha, astar, color="k", lw=2.0, label="BH trajectory")
    s4.scatter([alpha[0]], [astar[0]], color="green", s=80, zorder=5, label="Start")
    s4.scatter([alpha[-1]], [astar[-1]], color="red", s=80, zorder=5, label="End")
    s4.set_xlabel(r"$\alpha = GM\mu/\hbar c$")
    s4.set_ylabel(r"$\tilde{a}$")
    s4.set_title(r"Phase-space trajectory $(\alpha,\,\tilde{a})$")
    s4.legend(fontsize=7, ncol=2); s4.grid(True, alpha=0.25, linestyle="--")
    s4.set_xlim(0, results.alpha_0 * 1.15); s4.set_ylim(0, 1.0)

    fig2.suptitle(
        fr"Diagnostic plots — $m=l$ levels ($n \leq {results.max_n}$), "
        fr"$M_{{\rm BH}} = {results.M_BH_solar}\,M_\odot$, "
        fr"$\alpha_0 = {results.alpha_0}$, $\tilde{{a}}_0 = {results.a_star_0}$",
        fontsize=10, y=1.01)
    fig2.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig2.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Diagnostic figure saved to {save_path}")
    return fig2


# ══════════════════════════════════════════════════════════════════════
# Alpha sweep
# ══════════════════════════════════════════════════════════════════════

def run_alpha_sweep(M_BH_solar, a_star_0, max_n,
                    alpha_min=0.001, alpha_max=1.5, n_alpha_points=10,
                    output_dir=None):
    """
    Run the superradiance simulation over a geometric grid of alpha values
    and save the peak-characteristic tables for each point.

    Parameters
    ----------
    M_BH_solar : float
        Initial BH mass in solar masses (fixed across the sweep).
    a_star_0 : float
        Initial dimensionless spin (fixed across the sweep).
    max_n : int
        Maximum principal quantum number n (fixed across the sweep).
    alpha_min : float
        Lower bound of the alpha grid (inclusive).  Default: 0.001.
    alpha_max : float
        Upper bound of the alpha grid (inclusive).  Default: 1.5.
    n_alpha_points : int
        Number of alpha values sampled geometrically between alpha_min
        and alpha_max.  Default: 10.
    output_dir : str or None
        Directory in which to save the .dat files.  Defaults to a
        'Data/alpha_sweep' sub-folder next to this script.

    Returns
    -------
    list of SimulationResults (or None for failed points)
        One entry per alpha value, in the same order as the grid.
    """
    if output_dir is None:
        output_dir = os.path.join(_THIS_DIR, 'Data', 'alpha_sweep')
    os.makedirs(output_dir, exist_ok=True)

    alpha_grid = np.linspace(alpha_min, alpha_max, n_alpha_points)

    print("=" * 65)
    print(f"Alpha sweep: {n_alpha_points} points  "
          f"[{alpha_min:.4g}, {alpha_max:.4g}]  (geometric)")
    print(f"M_BH = {M_BH_solar} M_sun   ã₀ = {a_star_0}   max_n = {max_n}")
    print(f"Output dir: {output_dir}")
    print("=" * 65)

    all_results = []
    for i, alpha in enumerate(alpha_grid):
        print(f"\n── Sweep point {i+1}/{n_alpha_points}  α = {alpha:.6g} ──")
        results = run_simulation(
            M_BH_solar=M_BH_solar,
            a_star_0=a_star_0,
            alpha_0=alpha,
            max_n=max_n,
        )
        all_results.append(results)

        if results is None:
            print(f"  Skipped (no SR-active levels at α = {alpha:.6g}).")
            continue

        ann_char, tr_char = compute_peak_characteristics(results)
        print_peak_tables(ann_char, tr_char)
        save_peak_tables(ann_char, tr_char, results, output_dir=output_dir)

    # Summary
    n_ok = sum(r is not None for r in all_results)
    print("\n" + "=" * 65)
    print(f"Alpha sweep complete: {n_ok}/{n_alpha_points} points succeeded.")
    print("=" * 65)
    return all_results


# ══════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════

def main():
    # ── Physical input parameters — edit these ────────────────────────
    M_BH_SOLAR = 1e-6
<<<<<<< HEAD
    A_STAR_0   = 0.99
    ALPHA_0    = 0.42      # used for the single-run path only
    MAX_N      = 6

    # ── Alpha sweep parameters ────────────────────────────────────────
    RUN_ALPHA_SWEEP = False   # set True to run the sweep instead of a single sim
    N_ALPHA_POINTS  = 40      # number of geometrically-spaced alpha values
    ALPHA_MIN       = 0.01   # lower bound of the sweep
    ALPHA_MAX       = 0.8     # upper bound of the sweep
=======
    A_STAR_0   = 0.65
    ALPHA_0    = 0.15      # used for the single-run path only
    MAX_N      = 3

    # ── Alpha sweep parameters ────────────────────────────────────────
    RUN_ALPHA_SWEEP = False   # set True to run the sweep instead of a single sim
    N_ALPHA_POINTS  = 4      # number of geometrically-spaced alpha values
    ALPHA_MIN       = 0.2   # lower bound of the sweep
    ALPHA_MAX       = 0.3     # upper bound of the sweep
>>>>>>> 3b9e21da0657b2c765a7a216e3ca9d6366e9592c

    if RUN_ALPHA_SWEEP:
        run_alpha_sweep(
            M_BH_solar=M_BH_SOLAR,
            a_star_0=A_STAR_0,
            max_n=MAX_N,
            alpha_min=ALPHA_MIN,
            alpha_max=ALPHA_MAX,
            n_alpha_points=N_ALPHA_POINTS,
        )
        return

    # ── Single-alpha run ──────────────────────────────────────────────
    # Run simulation
    results = run_simulation(M_BH_solar=M_BH_SOLAR,
                             a_star_0=A_STAR_0,
                             alpha_0=ALPHA_0,
                             max_n=MAX_N,
                             tau_end_factor=1e10)
    if results is None:
        return

    # ── Compute and print peak characteristics ───────────────────────
    ann_char, tr_char = compute_peak_characteristics(results)
    print_peak_tables(ann_char, tr_char)

    # ── Save peak tables to Data folder ──────────────────────────────
    data_dir = os.path.join(_THIS_DIR, 'Data')
    save_peak_tables(ann_char, tr_char, results, output_dir=data_dir)

    # Save dimensionless (mass-independent) results
    save_dimensionless(results, filepath=os.path.join(data_dir,
        f"dimless_alpha{results.alpha_0:.3f}_a{results.a_star_0:.2f}.npz"))

    # Create output directory for plots
    plot_dir = os.path.join(_THIS_DIR, 'Plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Generate all plots
    main_path = os.path.join(plot_dir, 'superradiance_main.pdf')
    plot_occupations_and_spin(results, save_path=main_path)

    """gw_ann_path = os.path.join(plot_dir, 'superradiance_gw_strain.pdf')
<<<<<<< HEAD
    plot_gw_annihilation(results, save_path=gw_ann_path)
=======
    plot_gw_annihilation(results, save_path=gw_ann_path)"""
>>>>>>> 3b9e21da0657b2c765a7a216e3ca9d6366e9592c

    gw_tr_path = os.path.join(plot_dir, 'superradiance_gw_transitions.pdf')
    plot_gw_transitions(results, save_path=gw_tr_path)

    #ann_rates_path = os.path.join(plot_dir, 'annihilation_rates.pdf')
    #plot_annihilation_rates(results, save_path=ann_rates_path)

    diag_path = os.path.join(plot_dir, 'superradiance_diagnostics.pdf')
    plot_diagnostics(results, save_path=diag_path)"""

    plt.show()
    plt.close("all")

if __name__ == "__main__":
    main()