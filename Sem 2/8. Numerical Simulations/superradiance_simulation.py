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

"""

import os, sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Leaver module import ───────────────────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_LEAVER_DIR = os.path.join(_THIS_DIR, '..', '2. Relativistic Superradiance Rate')
sys.path.insert(0, _LEAVER_DIR if os.path.isdir(_LEAVER_DIR) else _THIS_DIR)

from leaver_superradiance import hydrogen_gamma


#------ Import ParamCalculator ----------------------------
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


# ══════════════════════════════════════════════════════════════════════
# Physical input parameters  — edit these
# ══════════════════════════════════════════════════════════════════════

M_BH_SOLAR = 1e-11    # Initial BH mass [solar masses]
A_STAR_0   = 0.65    # Initial dimensionless spin  a* = J/M²
ALPHA_0    = 0.1    # Gravitational coupling  α₀ = M₀μ
                     # Weak-coupling regime: hydrogenic rate valid for α ≲ 0.1


# ══════════════════════════════════════════════════════════════════════
# Derived internal quantities  (Planck units — do not edit)
# ══════════════════════════════════════════════════════════════════════

M0  = M_BH_SOLAR * M_SUN_PLANCK    # Initial BH mass [M_Pl]
MU  = ALPHA_0 / M0                  # Axion mass [M_Pl]
N0  = 1.0                           # Vacuum seed

# Physical time:  t [yr] = τ × TAU_TO_YR
#   Derivation: τ = t·μ → t [s] = τ/(μ·c²/ħ) = τ × M_BH_solar × GM_sun/c³ / α₀
TAU_TO_YR = M_BH_SOLAR * GM_SUN_OVER_C3 / (ALPHA_0 * YEAR_S)


# ══════════════════════════════════════════════════════════════════════
# Level list — all (n, l, m) with m = l, 2 ≤ n ≤ 8
# ══════════════════════════════════════════════════════════════════════
# These are the dominant SR modes. The m = l family has the fastest
# SR rates at given n because the azimuthal quantum number m is
# maximised for fixed l, maximising the SR window m·Ω_H.
# Excluded: m = 0 (never superradiant), m < l (slower rates).
#
# Total: Σ_{n=2}^{8} (n−1) = 1+2+3+4+5+6+7 = 28 levels
#
# Ordering: grouped by l, then ascending n within each l group.
# |211⟩, |311⟩…|811⟩, |322⟩…|822⟩, |433⟩…|833⟩, …, |877⟩

MAX_N = 7

LEVELS   = [(n, l, l) for l in range(1, MAX_N) for n in range(l + 1, MAX_N+1)]
N_LEVELS = len(LEVELS)   # 28

# Convenience arrays (fixed for the run)
_L_ARR = np.array([l for (n, l, m) in LEVELS])   # shape (28,)
_M_ARR = np.array([m for (n, l, m) in LEVELS])   # shape (28,)  (= _L_ARR)


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

    From the horizon area A = 16*pi*M_irr^2 and r+ = M*rhat_+, a = astar*M:

        M_irr = M * sqrt( (1 + sqrt(1 - astar^2)) / 2 )

    Properties
    ----------
    * M_irr <= M always; equality only for astar=0 (Schwarzschild).
    * Hawking area theorem: dM_irr/dt >= 0 for any classical process.
    * SR condition omega < m*Omega_H guarantees dM_irr > 0, verifiable
      from the first law:  dM = Omega_H*dJ + kappa/(8pi)*dA,  dA > 0.

    The rotational (extractable) energy is:  E_rot = M - M_irr
    The ODEs evolve (M, astar) together; M_irr is a derived quantity.
    """
    astar = float(np.clip(astar, 0.0, 0.99999))
    return M * np.sqrt((1.0 + np.sqrt(1.0 - astar**2)) / 2.0)


# ══════════════════════════════════════════════════════════════════════
# Superradiance rate  (hydrogenic / NR approximation)
# ══════════════════════════════════════════════════════════════════════

def gamma_tilde_sr(alpha, astar, n=2, l=1, m=1):
    """
    Dimensionless SR rate  Γ̃_SR = Γ_SR / μ  in the hydrogenic approximation.

    hydrogen_gamma(n, l, m, alpha, at) from leaver_superradiance.py returns
    Γ_NR in GM=c=1 units (same convention as the CF result: units of 1/M).
    Dividing by α = M·μ (with M=1 in GM=1 units) gives Γ̃ = Γ/μ.

        Γ̃_SR = hydrogen_gamma(n, l, m, α, a*) / α

    Returns 0.0 if α ≥ m·Ω_H (SR condition not met).
    Pure arithmetic — safe to call at every ODE step.
    """
    if alpha <= 0.0 or astar <= 0.0:
        return 0.0
    g = hydrogen_gamma(n, l, m, alpha, float(astar))
    return g / alpha if g > 0.0 else 0.0


# ══════════════════════════════════════════════════════════════════════
# Annihilation rate  (same-state: |nlm⟩ × |nlm⟩ → gg)
# ══════════════════════════════════════════════════════════════════════

_L_TO_SPEC = {1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h', 6: 'i', 7: 'k'}

def _level_string(n, l):
    """(n, l) → spectroscopic string expected by calc_annihilation_rate.
    e.g. (2,1) → '2p', (3,2) → '3d', (4,3) → '4f'. Returns None if unknown.
    """
    spec = _L_TO_SPEC.get(l)
    return f"{n}{spec}" if spec is not None else None


def _check_ann_available(n, l):
    """
    Test once at startup whether calc_annihilation_rate has a formula for (n,l).

    Always called with r_g=1, G_N=1 (GM=1 normalisation) to avoid overflow
    from large Planck-unit values.  Any exception → False (not yet computed).
    """
    lev_str = _level_string(n, l)
    if lev_str is None:
        return False
    try:
        omega_test = calc_omega_ann(r_g=1.0, alpha=0.1, n=n)
        with np.errstate(over='ignore', invalid='ignore'):
            val = calc_annihilation_rate(lev_str, alpha=0.1,
                                         omega=omega_test, G_N=1.0, r_g=1.0)
        return (val is not None
                and np.isfinite(float(val))
                and float(val) >= 0.0)
    except Exception:
        return False


# Built once at module load.  Guaranteed to always be defined: if the entire
# block fails for any reason, fall back to an all-False dict so that every
# downstream `_ANN_AVAILABLE.get(...)` call still works.
try:
    _ANN_AVAILABLE = {(n, l): _check_ann_available(n, l) for (n, l, m) in LEVELS}
    _ann_have    = [(n, l) for (n, l), v in _ANN_AVAILABLE.items() if     v]
    _ann_missing = [(n, l) for (n, l), v in _ANN_AVAILABLE.items() if not v]
    print(f"[annihilation] rates available for "
          f"{len(_ann_have)}/{len(_ANN_AVAILABLE)} levels")
    if _ann_missing:
        print(f"[annihilation] NOT yet computed: "
              + ", ".join(f"|{n}{_L_TO_SPEC.get(l,'?')}⟩"
                          for n, l in _ann_missing))
except Exception as _ann_err:
    print(f"[annihilation] WARNING: availability check failed ({_ann_err}). "
          f"Annihilation disabled.")
    _ANN_AVAILABLE = {(n, l): False for (n, l, m) in LEVELS}


def gamma_tilde_ann(n, l, Mtil):
    """
    Dimensionless annihilation rate  Γ̃_a = Γ_a / μ  for level (n, l, m=l).

    Convention: r_g = 1, G_N = 1  (GM=1 units, BH mass normalised to 1).
    Returned Γ_a is in units of 1/M_BH.  Conversion:

        Γ̃_a = Γ_a / (M_BH · μ) = Γ_a / (α₀ · M̃)

    Overflow warnings from ConvertedFunctions.py are suppressed; any
    NaN / Inf / negative result is returned as 0.
    """
    if not _ANN_AVAILABLE.get((n, l), False):
        return 0.0

    # alpha is dimensionless (α = M·μ), same in both modules
    alpha_cur = ALPHA_0 * Mtil

    # Convert BH mass to solar masses for calc_rg_from_bh_mass
    bh_mass_solar = Mtil * M_BH_SOLAR
    try:
        # r_g and omega in ParamCalculator use eV units
        r_g_ev = calc_rg_from_bh_mass(bh_mass_solar)
        omega_ev = calc_omega_ann(r_g=r_g_ev, alpha=alpha_cur, n=n)

        with np.errstate(over='ignore', invalid='ignore'):
            # Pass the physical Newton constant G_N from ParamCalculator
            gamma_a_ev = calc_annihilation_rate(
                _level_string(n, l),
                alpha = alpha_cur,
                omega = omega_ev,
                G_N   = G_N,
                r_g   = r_g_ev,
            )

        val = float(gamma_a_ev)
        # Gamma_a returned in eV (inverse time). Convert to dimensionless
        # rate Gamma_tilde = Gamma_a / mu where mu = alpha / r_g (eV).
        mu_ev = alpha_cur / r_g_ev
        if not np.isfinite(val) or val <= 0.0 or not np.isfinite(mu_ev) or mu_ev <= 0.0:
            return 0.0
        return val / mu_ev
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════
# ODE system  (multi-level, with annihilations)
# ══════════════════════════════════════════════════════════════════════

def odes(tau, y):
    """
    RHS of the multi-level BH-superradiance ODE system.

    State:  y = [lnN_0, …, lnN_{K-1},  M̃,  ã]
    Time:   τ = t·μ  (dimensionless)

    Equations
    ---------
      d(lnN_i)/dτ  =  Γ̃_SR_i  −  Γ̃_a_i · N_i

        Γ̃_SR drives exponential growth; Γ̃_a · N saturates it.
        Derived from dN/dt = Γ_SR N − Γ_a N² by dividing by N.

      dM̃/dτ  = −(α₀/M₀²) · Σ_i Γ̃_SR_i · N_i      [SR only]
      dã/dτ  =  Σ_i [Γ̃_SR_i N_i / (M₀² M̃²)]       [SR only]
                 · (2α·ã − m_i)

    The BH equations are unchanged by annihilation: that process acts on
    the already-formed cloud, radiating energy as GW that escapes to
    infinity, and does not directly alter M_BH or J_BH.
    """
    lnN_arr = y[:N_LEVELS]
    Mtil    = max(y[N_LEVELS],     1e-9)
    astar   = float(np.clip(y[N_LEVELS + 1], 0.0, 0.99999))

    N_arr = np.exp(lnN_arr)
    alpha = ALPHA_0 * Mtil

    # SR rates
    g_sr_arr = np.array([gamma_tilde_sr(alpha, astar, n, l, m)
                         for (n, l, m) in LEVELS])

    # Annihilation rates (0 for levels without computed formulae)
    g_ann_arr = np.array([gamma_tilde_ann(n, l, Mtil)
                          for (n, l, m) in LEVELS])

    # d(lnN_i)/dτ = Γ̃_SR_i − Γ̃_a_i · N_i
    dlnN_arr = g_sr_arr - g_ann_arr * N_arr

    # BH mass (SR only)
    dMtil = -(ALPHA_0 / M0**2) * np.dot(g_sr_arr, N_arr)

    # BH spin (SR only, each level weighted by m_i)
    dastar = (np.dot(g_sr_arr * N_arr, 2.0 * alpha * astar - _M_ARR)
              / (M0**2 * Mtil**2))

    return list(dlnN_arr) + [dMtil, dastar]


# ══════════════════════════════════════════════════════════════════════
# Terminal event
# ══════════════════════════════════════════════════════════════════════

def event_sr_off(tau, y):
    """
    Fires (terminal) when the last SR-active level switches off.

    This is whichever level has the highest m — for our level set
    (n ≤ 8, m = l) that is |877⟩ with m = 7.  Higher-m levels have
    a more permissive SR condition (α < m·ã/(2r̂₊)), so they remain
    SR-active down to lower spin and are the last to switch off.

    Each level's own SR condition is already enforced inside
    gamma_tilde_sr() via the (m·Ω_H − α) factor in hydrogen_gamma —
    levels stop growing automatically when their threshold is crossed.
    This event just tells the solver when to stop integrating entirely.

    Returns the m_max SR margin; goes negative when all levels are off.
    """
    Mtil  = max(y[N_LEVELS],     1e-9)
    astar = float(np.clip(y[N_LEVELS + 1], 0.0, 0.99999))
    m_max = int(_M_ARR.max())          # = 7 for n ≤ 8
    return sr_margin(ALPHA_0 * Mtil, astar, m=m_max)

event_sr_off.terminal  = True
event_sr_off.direction = -1


def event_occupation_decay(tau, y):
    """
    Terminal event: stop when the occupation number N drops below 1e10.
    """
    lnN = y[0]                     # only one level (index 0)
    N = np.exp(lnN)
    return N - 1e10                # crosses zero when N = 1e10

event_occupation_decay.terminal = True
event_occupation_decay.direction = -1   # trigger when decreasing through threshold

# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print(f"BH Superradiance — {N_LEVELS} levels (m=l, n≤8)  [hydrogenic]")
    print("=" * 65)
    print(f"  M_BH = {M_BH_SOLAR} M_sun  =  {M0:.3e} M_Pl")
    print(f"  ã₀   = {A_STAR_0}")
    print(f"  α₀   = {ALPHA_0}   →  μ = {MU:.3e} M_Pl")
    # Pre-compute initial SR rates for all levels — used in the level
    # table, the SR regime check, and the tau_end estimate.
    g0_all = {(n, l, m): gamma_tilde_sr(ALPHA_0, A_STAR_0, n, l, m)
              for (n, l, m) in LEVELS}

    print(f"\n  Levels tracked:")
    for n, l, m in LEVELS:
        g0_i   = g0_all[(n, l, m)]
        margin = sr_margin(ALPHA_0, A_STAR_0, m)
        if g0_i > 0:
            print(f"    |{n}{l}{m}⟩   Γ̃₀ = {g0_i:.3e}   "
                  f"τ_SR = {1/g0_i:.2e}   margin = {margin:.4f}")
        else:
            print(f"    |{n}{l}{m}⟩   Γ̃₀ = 0   (SR inactive, margin = {margin:.4f})")

    # ── Annihilation rates at initial parameters ──────────────────────
    print(f"\n  Annihilation rates at initial parameters:")
    for n, l, m in LEVELS:
        if _ANN_AVAILABLE.get((n, l), False):
            g_ann_0 = gamma_tilde_ann(n, l, 1.0)  
            if g_ann_0 > 0:
                print(f"    |{n}{l}{m}⟩   Γ̃_a = {g_ann_0:.3e}   "
                      f"τ_a = {1/g_ann_0:.2e}")
            else:
                print(f"    |{n}{l}{m}⟩   Γ̃_a = 0")
        else:
            print(f"    |{n}{l}{m}⟩   Γ̃_a = not computed")

        

    # ── SR regime check ──────────────────────────────────────────────
    # Identify which levels are SR-active at t=0 and find the fastest
    # one to anchor tau_end. If NO level is SR-active exit cleanly.

    active_t0 = [(n, l, m) for (n, l, m) in LEVELS if g0_all[(n, l, m)] > 0]

    if not active_t0:
        # No level satisfies the SR condition at the given (α₀, ã₀).
        # Print the SR margin for every level so the user can diagnose.
        print("\n  ✗  No levels are superradiant at the initial parameters.")
        print("     SR condition for level (n,l,m):  α < m·ã/(2r̂₊)")
        print(f"     Current:  α₀ = {ALPHA_0},  ã₀ = {A_STAR_0}")
        print(f"     Horizon factor:  ã/(2r̂₊) = {A_STAR_0/(2*rhat_plus(A_STAR_0)):.4f}")
        print("     SR margins at t=0 (must be > 0 for SR):")
        for n, l, m in LEVELS[:7]:   # show m=1 family as representative
            margin = sr_margin(ALPHA_0, A_STAR_0, m)
            print(f"       |{n}{l}{m}⟩  m·Ω_H − α = {margin:.4f}")
        print("\n  Increase ã₀ or decrease α₀ to enter the SR regime.")
        return

    # Fastest SR-active level at t=0 (anchors Phase 1 tau_end)
    dom_level = max(active_t0, key=lambda nlm: g0_all[nlm])
    dom_n, dom_l, dom_m = dom_level
    g0_dom = g0_all[dom_level]

    # ── Integration time ─────────────────────────────────────────────
    # tau_end covers two phases:
    #   Phase 1 — dominant level growth and BH spindown to ã_f
    #   Phase 2 — continued evolution of higher-m levels at ã_f
    #
    # Phase 2 only includes levels whose rate at ã_f is within RATE_RATIO
    # of the dominant rate.  Slower levels are cosmologically suppressed.

    RATE_RATIO = 1e40    # raise to include slower phase-2 levels; lower to
                          # shorten run time at large α

    N_sat   = M0**2 * A_STAR_0

    # ã after dominant level switches off: α = dom_m·ã/(2r̂₊)
    # → ã_f ≈ 2α/dom_m (small-α estimate, clamped to initial spin)
    astar_f = min(2.0 * ALPHA_0 * rhat_plus(A_STAR_0) / dom_m, A_STAR_0)

    tau_phase1 = 3.0 * np.log(max(N_sat, 2.0)) / g0_dom

    g_floor    = g0_dom / RATE_RATIO
    tau_phase2 = 0.0
    for n, l, m in LEVELS:
        if m <= dom_m:
            continue
        g_at_af = gamma_tilde_sr(ALPHA_0, astar_f, n, l, m)
        if g_at_af >= g_floor:
            tau_i      = np.log(max(N_sat, 2.0)) / g_at_af
            tau_phase2 = max(tau_phase2, tau_i)

    tau_end = 1e50# tau_phase1 + tau_phase2

    active_phase2 = [(n, l, m) for (n, l, m) in LEVELS
                     if m > dom_m
                     and gamma_tilde_sr(ALPHA_0, astar_f, n, l, m) >= g_floor]

    print(f"\n  Initially SR-active levels: {len(active_t0)}/{N_LEVELS}")
    print(f"  Fastest level: |{dom_n}{dom_l}{dom_m}⟩  "
          f"Γ̃₀ = {g0_dom:.4e}")
    print(f"  e-folding τ_SR = {1/g0_dom:.3e}  =  {1/g0_dom * TAU_TO_YR:.3e} yr")
    print(f"  Estimated ã after dominant switch-off: ã_f ≈ {astar_f:.4f}")
    if active_phase2:
        print(f"  Phase-2 levels (rate > {g_floor:.1e}): "
              + ", ".join(f"|{n}{l}{m}⟩" for n, l, m in active_phase2))
    else:
        print(f"  No phase-2 levels above rate floor {g_floor:.1e}"
              f" — higher modes frozen at this α")
    print(f"  τ_end = {tau_end:.3e}  =  {tau_end * TAU_TO_YR:.3e} yr")

    # ── Initial conditions ───────────────────────────────────────────
    # Every level seeded with one quantum from vacuum fluctuations
    y0 = [np.log(N0)] * N_LEVELS + [1.0, A_STAR_0]

    # ── Integrate ────────────────────────────────────────────────────
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
    lnN_all = sol.y[:N_LEVELS]          # shape (N_LEVELS, n_time)
    Mtil    = sol.y[N_LEVELS]           # shape (n_time,)
    astar   = sol.y[N_LEVELS + 1]       # shape (n_time,)

    # ── Derived quantities ───────────────────────────────────────────
    N_all   = np.exp(lnN_all)           # shape (N_LEVELS, n_time)
    alpha   = ALPHA_0 * Mtil
    M_BH    = Mtil * M0

    # Angular momentum: J_cloud = Σ_i m_i · N_i(t)
    J_cloud = _M_ARR @ N_all            # shape (n_time,)
    # Mass: M_cloud = μ · Σ_i N_i(t)
    M_cloud = MU * N_all.sum(axis=0)    # shape (n_time,)

    J_BH    = astar * Mtil**2 * M0**2
    J_total = J_BH + J_cloud
    dJ_frac = np.abs((J_total - J_total[0]) / J_total[0])
    t_yr    = tau * TAU_TO_YR

    # Christodoulou-Ruffini decomposition
    M_irr  = np.array([irreducible_mass(m, a) for m, a in zip(M_BH, astar)])
    E_rot  = M_BH - M_irr
    dM_irr = M_irr - M_irr[0]

    # SR rates over time for all levels  (N_LEVELS × n_time)
    g_sr_all = np.array([
        [gamma_tilde_sr(a, s, n, l, m) for a, s in zip(alpha, astar)]
        for (n, l, m) in LEVELS
    ])

    # Annihilation rates over time  (N_LEVELS × n_time)
    # Only evaluated where the formula is available; 0 elsewhere.
    g_ann_all = np.zeros((N_LEVELS, len(tau)))
    for k, (n, l, m) in enumerate(LEVELS):
        if _ANN_AVAILABLE.get((n, l), False):
            g_ann_all[k] = np.array([
                gamma_tilde_ann(n, l, Mt) for Mt in Mtil
            ])

    # ── GW strain from annihilation ──────────────────────────────────
    #
    # Formula:  h_i(t) = N_i(t) √[ 8 G_N Γ_a,i(t) / (r² ω_a,i(t)) ]
    #
    # In Planck units (G_N = 1):
    #   Γ_a,i^phys = Γ̃_a,i × μ                    [M_Pl]
    #   ω_a,i(t)   = 2μ (1 − α²(t)/2n²)            [M_Pl]
    #   r           = KPC_PLANCK                     [dimensionless, ℓ_Pl]
    #
    # ω_a,i is per-level (n-dependent) and slowly time-varying through α.
    # The GW is emitted at frequency f_GW = ω_a,i / (2π) in physical units.

    # omega_ann_all: annihilation frequency per level per time  (N_LEVELS × n_time)
    omega_ann_all = np.array([
        2.0 * MU * (1.0 - alpha**2 / (2.0 * n**2))
        for (n, l, m) in LEVELS
    ])                                                 # shape (N_LEVELS, n_time)

    # Physical annihilation rate  Γ_a^phys = Γ̃_a × μ
    Gamma_a_phys = g_ann_all * MU                      # shape (N_LEVELS, n_time)

    # --- Debug: print annihilation rate at peak N and at end ---
    peak_idx = np.argmax(N_all[0])  # assuming one level
    print(f"\nDebug: At peak N (t = {t_yr[peak_idx]:.2e} yr):")
    for k, (n, l, m) in enumerate(LEVELS):
        if g_ann_all[k].max() > 0:
            print(f"  |{n}{l}{m}⟩: Γ̃_a = {g_ann_all[k][peak_idx]:.3e}")
    print(f"At final time (t = {t_yr[-1]:.2e} yr):")
    for k, (n, l, m) in enumerate(LEVELS):
        if g_ann_all[k].max() > 0:
            print(f"  |{n}{l}{m}⟩: Γ̃_a = {g_ann_all[k][-1]:.3e}")

    # Strain: h = N √(8 Γ_a / (r² ω_a)), clamped to avoid sqrt of negative
    with np.errstate(invalid='ignore', divide='ignore'):
        h_all = N_all * np.sqrt(
            np.maximum(8.0 * Gamma_a_phys
                       / (KPC_PLANCK**2 * omega_ann_all), 0.0)
        )                                              # shape (N_LEVELS, n_time)
    h_all = np.nan_to_num(h_all, nan=0.0, posinf=0.0)

    # GW frequency per level (evaluated at initial α; slowly varying)
    f_gw_hz = np.array([
        2.0 * MU * (1.0 - ALPHA_0**2 / (2.0 * n**2)) * OMEGA_PL_TO_HZ
        for (n, l, m) in LEVELS
    ])                                                 # shape (N_LEVELS,)

    # GW power  P_GW = 2μ · Γ̃_a · N² (summed for diagnostic)
    P_GW_total = (2.0 * g_ann_all * N_all**2).sum(axis=0)

    # ── Summary ──────────────────────────────────────────────────────
    n_ann_active = sum(1 for (n, l, m) in LEVELS if _ANN_AVAILABLE.get((n, l), False))
    print(f"\nResults")
    print(f"  End:  τ = {tau[-1]:.4e}   t = {t_yr[-1]:.4f} yr")
    print(f"  ã   : {A_STAR_0:.4f} → {astar[-1]:.4f}  (Δã = {A_STAR_0 - astar[-1]:.4f})")
    print(f"  M   : {M_BH_SOLAR:.3f} → {Mtil[-1]*M_BH_SOLAR:.5f} M_sun"
          f"  (ΔM/M₀ = {1 - Mtil[-1]:.5f})")
    print(f"\n  Peak occupation numbers:")
    for k, (n, l, m) in enumerate(LEVELS):
        peak = lnN_all[k].max() / np.log(10)
        if peak > 0.5:
            ann_tag = ("[ann. active]"
                       if _ANN_AVAILABLE.get((n, l), False) else
                       "[ann. rate not computed]")
            print(f"    |{n}{l}{m}⟩   log₁₀N_max = {peak:.2f}   {ann_tag}")
    M_irr0_sol = M_irr[0]  / M_SUN_PLANCK
    M_irrf_sol = M_irr[-1] / M_SUN_PLANCK
    E_rot0_sol = E_rot[0]  / M_SUN_PLANCK
    E_rotf_sol = E_rot[-1] / M_SUN_PLANCK
    print(f"\n  Christodoulou-Ruffini decomposition:")
    print(f"  M_irr : {M_irr0_sol:.5e} → {M_irrf_sol:.5e} M_sun"
          f"  (ΔM_irr = +{M_irrf_sol - M_irr0_sol:.5e}, area theorem ✓)")
    print(f"  E_rot : {E_rot0_sol:.5e} → {E_rotf_sol:.5e} M_sun")
    print(f"  M_cloud total : {M_cloud[-1]/M_SUN_PLANCK:.5e} M_sun")
    print(f"  Annihilation  : {n_ann_active}/{N_LEVELS} levels have computed rates")
    h_peaks = [(h_all[k].max(), n, l, m) for k,(n,l,m) in enumerate(LEVELS)
               if h_all[k].max() > 0]
    if h_peaks:
        print(f"  Peak GW strains (at 1 kpc):")
        for h_pk, n, l, m in sorted(h_peaks, reverse=True)[:5]:
            print(f"    |{n}{l}{m}⟩   h_max = {h_pk:.3e}"
                  f"   f_GW ≈ {f_gw_hz[LEVELS.index((n,l,m))]:.3e} Hz")
    if n_ann_active > 0:
        print(f"  J conservation: not exact — GW carries angular momentum (expected)")
    else:
        print(f"  J conservation: {dJ_frac.max():.2e}  ✓  (no ann. rates active)")
    print(f"  Area theorem  : {'✓' if dM_irr.min() >= -1e-6*M_irr[0] else '✗'}")

    # ── LaTeX rendering ───────────────────────────────────────────────
    # log₁₀ N for plotting  (calculations remain in natural log throughout)
    log10N_all = lnN_all / np.log(10)   # shape (N_LEVELS, n_time)

    plt.rcParams.update({
        "text.usetex":          True,
        "font.family":          "serif",
        "font.serif":           ["Computer Modern Roman"],
        "text.latex.preamble":  r"\usepackage{amsmath}",
    })

    # ── Colour scheme: one colour per l value (7 colours) ────────────
    # Within each l group, lines become progressively lighter for higher n.
    # l=1…7 mapped to a qualitative palette.
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

    def level_style(n, l, m):
        """Return (color, linestyle, alpha, linewidth) for level (n,l,m)."""
        base_col = _L_COLORS[l]
        # Within the l-group, n goes from l+1 upward.
        # Use progressively slightly thinner/lighter lines for higher n.
        n_idx    = n - (l + 1)          # 0 for lowest n in this l-group
        ls       = _LSTYLES[n_idx % len(_LSTYLES)]
        lw       = max(2.0 - 0.2 * n_idx, 0.8)
        return base_col, ls, lw

    # ══════════════════════════════════════════════════════════════════
    # Main plot: spin panel (top) + log₁₀N panel (bottom)
    # ══════════════════════════════════════════════════════════════════
    color_spin = "firebrick"

    fig_main = plt.figure(figsize=(8, 6))
    gs_main  = GridSpec(
        2, 1, figure=fig_main,
        height_ratios=[0.5, 3.5],
        hspace=0.1,
    )
    ax_spin = fig_main.add_subplot(gs_main[0])
    ax_main = fig_main.add_subplot(gs_main[1], sharex=ax_spin)

    # ── Top panel: BH spin ã(t) ──────────────────────────────────────
    ax_spin.plot(t_yr, astar, color='k', lw=1.2)
    ax_spin.set_ylabel(r"$\tilde{a}$", fontsize=11)
    ax_spin.tick_params(axis="y", labelsize=9)
    ax_spin.tick_params(axis="x", labelbottom=False)
    ax_spin.set_ylim(-0.05, 1.10)
    ax_spin.set_yticks([0.0, 0.5, 1.0])
    ax_spin.grid(True, alpha=0.25, linestyle="--")
    ax_spin.axhline(astar[-1], ls=":", color=color_spin, lw=0.8, alpha=0.55)
    ax_spin.annotate(
        fr"$\tilde{{a}}_f = {astar[-1]:.3f}$",
        xy=(t_yr[len(t_yr)//2], astar[-1]),
        xytext=(0, 5), textcoords="offset points",
        fontsize=8, color=color_spin,
    )
    ax_spin.annotate(
        fr"$\tilde{{a}}_0 = {A_STAR_0}$",
        xy=(t_yr[0], A_STAR_0),
        xytext=(6, -10), textcoords="offset points",
        fontsize=8, color=color_spin,
    )
    ax_spin.set_xscale('log')
    
    # ── Bottom panel: log₁₀N(t) for all levels ───────────────────────
    for k, (n, l, m) in enumerate(LEVELS):
        col, ls, lw = level_style(n, l, m)
        label = rf"$|{n}{l}{m}\rangle$"
        # Only add a legend entry for levels that grow noticeably
        show_label = log10N_all[k].max() > 0.5
        ax_main.plot(t_yr, log10N_all[k],
                     color=col, ls=ls, lw=lw,
                     label=label if show_label else "_nolegend_")

    ax_main.set_xlabel(r"$t$  [yr]", fontsize=12)
    ax_main.set_ylabel(r"$\log_{10}\, N$", fontsize=12)
    ax_main.tick_params(axis="y")
    ax_main.grid(True, alpha=0.25, linestyle="--")
    ax_main.legend(fontsize=9, loc="upper left", ncol=2)
    ax_main.set_ylim(0, log10N_all.max() * 1.1)
    t_min_pos_main = t_yr[t_yr > 0][0] if np.any(t_yr > 0) else t_yr[1]
    ax_main.set_xlim(t_min_pos_main, t_yr[-1])

    ax_main.set_xscale('log')

    main_path    = os.path.join(_THIS_DIR, 'Plots', 'superradiance_main.pdf')
    main_preview = "Sem 2/8. Numerical Simulations/Plots/superradiance_main.png"
    os.makedirs(os.path.dirname(main_path), exist_ok=True)
    fig_main.savefig(main_path,    dpi=150, bbox_inches="tight")
    fig_main.savefig(main_preview, dpi=150, bbox_inches="tight")
    print(f"\nMain figure saved to {main_path}")

    # ══════════════════════════════════════════════════════════════════
    # GW strain plot: spin panel (top) + h(t) per level (bottom)
    # Same layout as the occupation number main plot.
    # Only levels with computed annihilation rates are drawn.
    # ══════════════════════════════════════════════════════════════════
    has_strain = any(h_all[k].max() > 0 for k in range(N_LEVELS))

    if has_strain:
        fig_gw = plt.figure(figsize=(8, 6))
        gs_gw  = GridSpec(2, 1, figure=fig_gw,
                          height_ratios=[0.5, 3.5], hspace=0.1)
        ax_gw_spin = fig_gw.add_subplot(gs_gw[0])
        ax_gw_h    = fig_gw.add_subplot(gs_gw[1], sharex=ax_gw_spin)

        # ── Top panel: BH spin (identical style to main plot) ────────
        ax_gw_spin.plot(t_yr, astar, color='k', lw=1.2)
        ax_gw_spin.set_ylabel(r"$\tilde{a}$", fontsize=11)
        ax_gw_spin.tick_params(axis="y", labelsize=9)
        ax_gw_spin.tick_params(axis="x", labelbottom=False)
        ax_gw_spin.set_ylim(-0.05, 1.10)
        ax_gw_spin.set_yticks([0.0, 0.5, 1.0])
        ax_gw_spin.grid(True, alpha=0.25, linestyle="--")
        ax_gw_spin.axhline(astar[-1], ls=":", color=color_spin, lw=0.8, alpha=0.55)
        ax_gw_spin.annotate(
            fr"$\tilde{{a}}_f = {astar[-1]:.3f}$",
            xy=(t_yr[len(t_yr)//2], astar[-1]),
            xytext=(0, 5), textcoords="offset points",
            fontsize=8, color=color_spin,
        )
        ax_gw_spin.annotate(
            fr"$\tilde{{a}}_0 = {A_STAR_0}$",
            xy=(t_yr[0], A_STAR_0),
            xytext=(6, -10), textcoords="offset points",
            fontsize=8, color=color_spin,
        )
        ax_gw_spin.set_xscale('log')

        # ── Bottom panel: strain h(t) per level ──────────────────────
        for k, (n, l, m) in enumerate(LEVELS):
            if h_all[k].max() <= 0:
                continue                    # no annihilation rate — skip
            col, ls, lw = level_style(n, l, m)
            f_hz = f_gw_hz[k]
            # Format frequency label  (Hz, kHz, MHz, GHz)
            if   f_hz >= 1e9:  f_str = fr"$f={f_hz/1e9:.1f}$\,GHz"
            elif f_hz >= 1e6:  f_str = fr"$f={f_hz/1e6:.1f}$\,MHz"
            elif f_hz >= 1e3:  f_str = fr"$f={f_hz/1e3:.1f}$\,kHz"
            else:               f_str = fr"$f={f_hz:.1f}$\,Hz"
            label = rf"$|{n}{l}{m}\rangle$\;({f_str})"
            ax_gw_h.semilogy(t_yr, np.maximum(h_all[k], 1e-100),
                             color=col, ls=ls, lw=lw, label=label)

        ax_gw_h.set_xlabel(r"$t$  [yr]", fontsize=12)
        ax_gw_h.set_ylabel(r"$h$ (at 1\,kpc)", fontsize=12)
        ax_gw_h.grid(True, alpha=0.25, which="both", linestyle="--")
        ax_gw_h.legend(fontsize=8, loc="upper left", ncol=1)
        ax_gw_h.set_xscale('log')
        ax_gw_h.set_xlim(t_min_pos_main, t_yr[-1])

        fig_gw.suptitle(
            fr"GW strain from axion annihilation — $m=l$ levels ($n\leq 8$),  "
            fr"$M_{{\rm BH}} = {M_BH_SOLAR}\,M_\odot$,  "
            fr"$\alpha_0 = {ALPHA_0}$,  distance $= 1\,\rm kpc$",
            fontsize=9, y=1.01,
        )

        gw_path    = os.path.join(_THIS_DIR, 'Plots', 'superradiance_gw_strain.pdf')
        gw_preview = "Sem 2/8. Numerical Simulations/Plots/superradiance_gw_strain.png"
        fig_gw.savefig(gw_path,    dpi=150, bbox_inches="tight")
        fig_gw.savefig(gw_preview, dpi=150, bbox_inches="tight")
        print(f"GW strain figure saved to {gw_path}")
    else:
        print("GW strain plot skipped — no annihilation rates available for active levels.")


    
    # ══════════════════════════════════════════════════════════════════════
    # Additional diagnostic: annihilation rates over time
    # ══════════════════════════════════════════════════════════════════════
    if np.any(g_ann_all > 0):
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        for k, (n, l, m) in enumerate(LEVELS):
            if g_ann_all[k].max() > 0:
                col, ls, lw = level_style(n, l, m)
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
        
        ann_path = os.path.join(_THIS_DIR, 'Plots', 'annihilation_rates.pdf')
        os.makedirs(os.path.dirname(ann_path), exist_ok=True)
        fig3.savefig(ann_path, dpi=150, bbox_inches="tight")
        print(f"Annihilation rates plot saved to {ann_path}")
    else:
        print("No positive annihilation rates found in the simulation.")

    # ══════════════════════════════════════════════════════════════════
    # Secondary plot: SR rates, energy decomposition, J budget, phase space
    # ══════════════════════════════════════════════════════════════════
    fig2 = plt.figure(figsize=(13, 9))
    gs2  = GridSpec(2, 2, figure=fig2, hspace=0.42, wspace=0.34)
    s1 = fig2.add_subplot(gs2[0, 0])
    s2 = fig2.add_subplot(gs2[0, 1])
    s3 = fig2.add_subplot(gs2[1, 0])
    s4 = fig2.add_subplot(gs2[1, 1])
    tl = r"$t$  [yr]"

    # For log x-axis: t_yr[0] = 0 is undefined in log scale.
    # Use the first strictly positive time point as the left limit.
    t_min_pos = t_yr[t_yr > 0][0] if np.any(t_yr > 0) else t_yr[1]
    t_max     = t_yr[-1]

    # S1. SR rates — all levels
    for k, (n, l, m) in enumerate(LEVELS):
        col, ls, lw = level_style(n, l, m)
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

    # S2. Christodoulou-Ruffini energy decomposition
    s2.plot(t_yr, M_irr   / M_SUN_PLANCK, color="firebrick",  lw=1.8,
            label=r"$M_{\rm irr}$")
    s2.plot(t_yr, E_rot   / M_SUN_PLANCK, color="darkorange", lw=1.8,
            label=r"$E_{\rm rot}$")
    s2.plot(t_yr, M_cloud / M_SUN_PLANCK, color="royalblue",  lw=1.8, ls="--",
            label=r"$M_{\rm cloud}$")
    s2.axhline(E_rot[0] / M_SUN_PLANCK, ls=":", color="darkorange",
               lw=0.9, alpha=0.5)
    s2.set_xlabel(tl); s2.set_ylabel(r"Mass-energy  $[M_\odot]$")
    s2.set_title("Christodoulou--Ruffini decomposition")
    s2.legend(fontsize=9)
    s2.set_xscale('log')
    s2.set_xlim(t_min_pos, t_max)
    s2.grid(True, alpha=0.25, linestyle="--")

    # S3. Angular momentum budget
    J0 = J_total[0]
    s3.plot(t_yr, J_BH    / J0, color="firebrick", lw=1.8,
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

    # S4. Phase-space trajectory (α, ã)
    # Show SR boundaries for each distinct m value
    a_line = np.linspace(0.01, 0.9999, 500)
    for m_val in sorted(set(_M_ARR)):
        alpha_bnd = m_val * a_line / (2.0 * (1.0 + np.sqrt(1.0 - a_line**2)))
        col = _L_COLORS[m_val]
        s4.plot(alpha_bnd, a_line, ls="--", lw=1.0, color=col, alpha=0.7,
                label=rf"$m={m_val}$ boundary")

    s4.plot(alpha, astar, color="k", lw=2.0, label="BH trajectory")
    s4.scatter([alpha[0]],  [astar[0]],  color="green", s=80, zorder=5,
               label="Start")
    s4.scatter([alpha[-1]], [astar[-1]], color="red",   s=80, zorder=5,
               label="End")
    s4.set_xlabel(r"$\alpha = GM\mu/\hbar c$")
    s4.set_ylabel(r"$\tilde{a}$")
    s4.set_title(r"Phase-space trajectory $(\alpha,\,\tilde{a})$")
    s4.legend(fontsize=7, ncol=2); s4.grid(True, alpha=0.25, linestyle="--")
    s4.set_xlim(0, ALPHA_0 * 1.15); s4.set_ylim(0, 1.0)

    fig2.suptitle(
        fr"Diagnostic plots — $m=l$ levels ($n \leq 8$),  "
        fr"$M_{{\rm BH}} = {M_BH_SOLAR}\,M_\odot$,  "
        fr"$\alpha_0 = {ALPHA_0}$,  $\tilde{{a}}_0 = {A_STAR_0}$",
        fontsize=10, y=1.01,
    )
    fig2.tight_layout()

    sec_path    = os.path.join(_THIS_DIR, 'Plots', 'superradiance_diagnostics.pdf')
    sec_preview = "Sem 2/8. Numerical Simulations/Plots/superradiance_diagnostics.png"
    fig2.savefig(sec_path,    dpi=150, bbox_inches="tight")
    fig2.savefig(sec_preview, dpi=150, bbox_inches="tight")
    print(f"Diagnostic figure saved to {sec_path}")

    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()