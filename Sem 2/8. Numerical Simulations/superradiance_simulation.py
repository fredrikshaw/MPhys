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
    h_all: np.ndarray           # (N_LEVELS, n_time)
    f_gw_hz: np.ndarray         # (N_LEVELS,)
    P_GW_total: np.ndarray      # (n_time,)

    # GW strain from transitions
    g_tr_matrix: np.ndarray     # (n_tr_pairs, n_time)
    h_tr_matrix: np.ndarray     # (n_tr_pairs, n_time)
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
        lnN_arr = y[:N_LEVELS]
        Mtil    = max(y[N_LEVELS],     1e-9)
        astar   = float(np.clip(y[N_LEVELS + 1], 0.0, 0.99999))

        # After
        N_arr = np.exp(np.clip(lnN_arr, None, 700.0))
        alpha = alpha_0 * Mtil

        # SR rates
        g_sr_arr = np.array([gamma_tilde_sr(alpha, astar, n, l, m)
                             for (n, l, m) in LEVELS])

        # Annihilation rates
        g_ann_arr = np.array([gamma_tilde_ann(n, l, Mtil)
                              for (n, l, m) in LEVELS])

        # Transition contributions
        dlnN_tr = np.zeros(N_LEVELS)
        for idx_i, idx_j, n_i, l_i, n_j, l_j in TR_PAIRS_IDX:
            g_tr = gamma_tilde_tr(n_i, l_i, n_j, l_j, Mtil)
            if g_tr > 0.0:
                dlnN_tr[idx_i] -= g_tr * N_arr[idx_j]
                dlnN_tr[idx_j] += g_tr * N_arr[idx_i]

        # d(lnN_i)/dτ
        dlnN_arr = g_sr_arr - g_ann_arr * N_arr + dlnN_tr

        # BH mass (SR only)
        dMtil = -(alpha_0 / M0**2) * np.dot(g_sr_arr, N_arr)

        # BH spin (SR only)
        dastar = (np.dot(g_sr_arr * N_arr, 2.0 * alpha * astar - _M_ARR)
                  / (M0**2 * Mtil**2))

        return list(dlnN_arr) + [dMtil, dastar]

    # ── Terminal events ──────────────────────────────────────────────
    def event_sr_off(tau, y):
        Mtil  = max(y[N_LEVELS],     1e-9)
        astar = float(np.clip(y[N_LEVELS + 1], 0.0, 0.99999))
        m_max = int(_M_ARR.max())
        return sr_margin(alpha_0 * Mtil, astar, m=m_max)
    event_sr_off.terminal  = True
    event_sr_off.direction = -1

    def event_occupation_decay(tau, y):
        lnN = y[0]   # stop if the first level's N drops too low
        N = np.exp(lnN)
        return N - 1e10
    event_occupation_decay.terminal = True
    event_occupation_decay.direction = -1

    # ── Initial SR analysis and integration time ──────────────────────
    g0_all = {(n, l, m): gamma_tilde_sr(alpha_0, a_star_0, n, l, m)
              for (n, l, m) in LEVELS}
    active_t0 = [(n, l, m) for (n, l, m) in LEVELS if g0_all[(n, l, m)] > 0]

    if not active_t0:
        print("\n  ✗  No levels are superradiant at the initial parameters.")
        print("     SR margins at t=0:")
        for n, l, m in LEVELS[:7]:
            margin = sr_margin(alpha_0, a_star_0, m)
            print(f"       |{n}{l}{m}⟩  m·Ω_H − α = {margin:.4f}")
        print("\n  Increase ã₀ or decrease α₀ to enter the SR regime.")
        return None

    # Fastest SR-active level at t=0
    dom_level = max(active_t0, key=lambda nlm: g0_all[nlm])
    dom_n, dom_l, dom_m = dom_level
    g0_dom = g0_all[dom_level]

    # ── Time span estimation ──────────────────────────────────────────
    RATE_RATIO = 1e40
    N_sat = M0**2 * a_star_0
    astar_f = min(2.0 * alpha_0 * rhat_plus(a_star_0) / dom_m, a_star_0)
    tau_phase1 = 3.0 * np.log(max(N_sat, 2.0)) / g0_dom

    g_floor = g0_dom / RATE_RATIO
    tau_phase2 = 0.0
    for n, l, m in LEVELS:
        if m <= dom_m:
            continue
        g_at_af = gamma_tilde_sr(alpha_0, astar_f, n, l, m)
        if g_at_af >= g_floor:
            tau_i = np.log(max(N_sat, 2.0)) / g_at_af
            tau_phase2 = max(tau_phase2, tau_i)

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
        g0_i = g0_all[(n, l, m)]
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
    active_phase2 = [(n,l,m) for (n,l,m) in LEVELS
                     if m>dom_m and gamma_tilde_sr(alpha_0, astar_f, n, l, m) >= g_floor]
    print(f"\n  Initially SR-active levels: {len(active_t0)}/{N_LEVELS}")
    print(f"  Fastest level: |{dom_n}{dom_l}{dom_m}⟩  Γ̃₀ = {g0_dom:.4e}")
    print(f"  e-folding τ_SR = {1/g0_dom:.3e}  =  {1/g0_dom * TAU_TO_YR:.3e} yr")
    print(f"  Estimated ã after dominant switch-off: ã_f ≈ {astar_f:.4f}")
    if active_phase2:
        print(f"  Phase-2 levels: " + ", ".join(f"|{n}{l}{m}⟩" for n,l,m in active_phase2))
    else:
        print(f"  No phase-2 levels above rate floor {g_floor:.1e}")
    print(f"  τ_end = {tau_end:.3e}  =  {tau_end * TAU_TO_YR:.3e} yr")

    # ── Initial conditions and integration ───────────────────────────
    y0 = [np.log(N0)] * N_LEVELS + [1.0, a_star_0]
    print("\nIntegrating ... ", end="", flush=True)
    sol = solve_ivp(
        odes,
        t_span       = (0.0, tau_end),
        y0           = y0,
        method       = "RK45",
        events       = [event_occupation_decay],
        rtol         = 1e-9,
        atol         = 1e-12,
        dense_output = False,
    )
    print("done.")

    tau   = sol.t
    lnN_all = sol.y[:N_LEVELS]
    Mtil    = sol.y[N_LEVELS]
    astar   = sol.y[N_LEVELS + 1]

    # ── Derived quantities ────────────────────────────────────────────
    N_all   = np.exp(lnN_all)
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
        tau=tau,
        lnN_all=lnN_all,
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
        f_gw_hz=f_gw_hz,
        P_GW_total=P_GW_total,
        g_tr_matrix=g_tr_matrix,
        h_tr_matrix=h_tr_matrix,
        f_tr_hz=f_tr_hz,
        astar_f=astar_f,
        t_final_yr=t_yr[-1],
        active_levels_t0=len(active_t0),
        dominant_level=dom_level,
    )
    return res


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
    """Main plot: spin panel (top) + log10 N(t) for all levels (bottom)."""
    if results is None:
        print("No simulation results to plot.")
        return
    t_yr = results.t_yr
    astar = results.astar
    log10N_all = results.lnN_all / np.log(10)
    LEVELS = results.LEVELS

    color_spin = "firebrick"

    fig_main = plt.figure(figsize=(8, 6))
    gs_main = GridSpec(2, 1, figure=fig_main, height_ratios=[0.5, 3.5], hspace=0.1)
    ax_spin = fig_main.add_subplot(gs_main[0])
    ax_main = fig_main.add_subplot(gs_main[1], sharex=ax_spin)

    # Spin panel
    ax_spin.plot(t_yr, astar, color='k', lw=1.2)
    ax_spin.set_ylabel(r"$\tilde{a}$", fontsize=11)
    ax_spin.tick_params(axis="y", labelsize=9)
    ax_spin.tick_params(axis="x", labelbottom=False)
    ax_spin.set_ylim(-0.05, 1.10)
    ax_spin.set_yticks([0.0, 0.5, 1.0])
    ax_spin.grid(True, alpha=0.25, linestyle="--")
    ax_spin.axhline(astar[-1], ls=":", color=color_spin, lw=0.8, alpha=0.55)
    ax_spin.annotate(fr"$\tilde{{a}}_f = {astar[-1]:.3f}$",
                     xy=(t_yr[len(t_yr)//2], astar[-1]),
                     xytext=(0, 5), textcoords="offset points",
                     fontsize=8, color=color_spin)
    ax_spin.annotate(fr"$\tilde{{a}}_0 = {results.a_star_0}$",
                     xy=(t_yr[0], results.a_star_0),
                     xytext=(6, -10), textcoords="offset points",
                     fontsize=8, color=color_spin)
    ax_spin.set_xscale('log')

    # Occupation panel
    for k, (n, l, m) in enumerate(LEVELS):
        col, ls, lw = level_style(n, l)
        label = rf"$|{n}{l}{m}\rangle$"
        show_label = log10N_all[k].max() > 0.5
        ax_main.plot(t_yr, log10N_all[k], color=col, ls=ls, lw=lw,
                     label=label if show_label else "_nolegend_")
    ax_main.set_xlabel(r"$t$  [yr]", fontsize=12)
    ax_main.set_ylabel(r"$\log_{10}\, N$", fontsize=12)
    ax_main.grid(True, alpha=0.25, linestyle="--")
    ax_main.legend(fontsize=9, loc="upper left", ncol=2)
    ax_main.set_ylim(0, log10N_all.max() * 1.1)
    t_min_pos = t_yr[t_yr > 0][0] if np.any(t_yr > 0) else t_yr[1]
    ax_main.set_xlim(t_min_pos, t_yr[-1])
    ax_main.set_xscale('log')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig_main.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Main figure saved to {save_path}")
    return fig_main


def plot_gw_annihilation(results: SimulationResults, save_path=None):
    """GW strain from annihilation: spin + h(t) per level."""
    if results is None:
        return
    t_yr = results.t_yr
    astar = results.astar
    h_all = results.h_all
    f_gw_hz = results.f_gw_hz
    LEVELS = results.LEVELS
    if not np.any(h_all > 0):
        print("No annihilation GW strain to plot.")
        return

    color_spin = "firebrick"
    fig_gw = plt.figure(figsize=(8, 6))
    gs_gw = GridSpec(2, 1, figure=fig_gw, height_ratios=[0.5, 3.5], hspace=0.1)
    ax_gw_spin = fig_gw.add_subplot(gs_gw[0])
    ax_gw_h = fig_gw.add_subplot(gs_gw[1], sharex=ax_gw_spin)

    ax_gw_spin.plot(t_yr, astar, color='k', lw=1.2)
    ax_gw_spin.set_ylabel(r"$\tilde{a}$", fontsize=11)
    ax_gw_spin.tick_params(axis="y", labelsize=9)
    ax_gw_spin.tick_params(axis="x", labelbottom=False)
    ax_gw_spin.set_ylim(-0.05, 1.10)
    ax_gw_spin.set_yticks([0.0, 0.5, 1.0])
    ax_gw_spin.grid(True, alpha=0.25, linestyle="--")
    ax_gw_spin.axhline(astar[-1], ls=":", color=color_spin, lw=0.8, alpha=0.55)
    ax_gw_spin.set_xscale('log')

    for k, (n, l, m) in enumerate(LEVELS):
        if h_all[k].max() <= 0:
            continue
        col, ls, lw = level_style(n, l)
        f_hz = f_gw_hz[k]
        if f_hz >= 1e9: f_str = fr"$f={f_hz/1e9:.1f}$\,GHz"
        elif f_hz >= 1e6: f_str = fr"$f={f_hz/1e6:.1f}$\,MHz"
        elif f_hz >= 1e3: f_str = fr"$f={f_hz/1e3:.1f}$\,kHz"
        else: f_str = fr"$f={f_hz:.1f}$\,Hz"
        label = rf"$|{n}{l}{m}\rangle$\;({f_str})"
        ax_gw_h.semilogy(t_yr, np.maximum(h_all[k], 1e-100),
                         color=col, ls=ls, lw=lw, label=label)
    ax_gw_h.set_xlabel(r"$t$  [yr]", fontsize=12)
    ax_gw_h.set_ylabel(r"$h$ (at 1\,kpc)", fontsize=12)
    ax_gw_h.grid(True, alpha=0.25, which="both", linestyle="--")
    ax_gw_h.legend(fontsize=8, loc="upper left", ncol=1)
    ax_gw_h.set_xscale('log')
    t_min_pos = t_yr[t_yr > 0][0] if np.any(t_yr > 0) else t_yr[1]
    ax_gw_h.set_xlim(t_min_pos, t_yr[-1])

    fig_gw.suptitle(
        fr"GW strain from axion annihilation — $m=l$ levels ($n\leq{results.max_n}$), "
        fr"$M_{{\rm BH}} = {results.M_BH_solar}\,M_\odot$, "
        fr"$\alpha_0 = {results.alpha_0}$, distance $= 1\,\rm kpc$",
        fontsize=9, y=1.01)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig_gw.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"GW annihilation strain figure saved to {save_path}")
    return fig_gw


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
    A_STAR_0   = 0.65
    ALPHA_0    = 0.15      # used for the single-run path only
    MAX_N      = 8

    # ── Alpha sweep parameters ────────────────────────────────────────
    RUN_ALPHA_SWEEP = True   # set True to run the sweep instead of a single sim
    N_ALPHA_POINTS  = 25      # number of geometrically-spaced alpha values
    ALPHA_MIN       = 0.3   # lower bound of the sweep
    ALPHA_MAX       = 0.8     # upper bound of the sweep

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
                             max_n=MAX_N)
    if results is None:
        return

    # ── Compute and print peak characteristics ───────────────────────
    ann_char, tr_char = compute_peak_characteristics(results)
    print_peak_tables(ann_char, tr_char)

    # ── Save peak tables to Data folder ──────────────────────────────
    data_dir = os.path.join(_THIS_DIR, 'Data')
    save_peak_tables(ann_char, tr_char, results, output_dir=data_dir)

    # Create output directory for plots
    plot_dir = os.path.join(_THIS_DIR, 'Plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Generate all plots
    main_path = os.path.join(plot_dir, 'superradiance_main.pdf')
    plot_occupations_and_spin(results, save_path=main_path)

    gw_ann_path = os.path.join(plot_dir, 'superradiance_gw_strain.pdf')
    plot_gw_annihilation(results, save_path=gw_ann_path)

    gw_tr_path = os.path.join(plot_dir, 'superradiance_gw_transitions.pdf')
    plot_gw_transitions(results, save_path=gw_tr_path)

    #ann_rates_path = os.path.join(plot_dir, 'annihilation_rates.pdf')
    #plot_annihilation_rates(results, save_path=ann_rates_path)

    diag_path = os.path.join(plot_dir, 'superradiance_diagnostics.pdf')
    plot_diagnostics(results, save_path=diag_path)

    plt.show()
    plt.close("all")

if __name__ == "__main__":
    main()