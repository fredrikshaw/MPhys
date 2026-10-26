"""
======================================================================
Black Hole Superradiance Simulation — Multi-level (n ≤ 8, m = l)
======================================================================
Simulates the coupled growth of all superradiant axion cloud levels
with m = l (the dominant SR family) up to n = 8, and the resulting
Kerr BH spin-down with shared backreaction across all levels.

The ODE evolves epsilon=N/(G*M_initial**2), q=M/M_initial and spin in
tau=mu*t, using a fixed dimensionless seed. No physical PBH mass is
required. Save coupling-grid solutions, then use to_physical() or
rescale_to_mass() to obtain occupations, strains and times at any mass.

Refactored: simulation run, result storage, plotting, peak analysis,
and data export are separated into dedicated functions.
"""

import os, sys
import csv
import time
from functools import lru_cache
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
HBAR_EV_S      = 6.582119569e-16  # hbar [eV s]
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
class DimensionlessResults:
    """Universal solution at fixed (alpha_0, a_star_0, max_n, epsilon_seed).

    tau = mu*t, Mtil = M/M_initial, eps = N/(G*M_initial**2).
    ann_coeff_all and tr_coeff_matrix are G*M_initial**2 * Gamma/mu,
    whereas g_sr_all is Gamma_sr/mu. No physical BH mass is stored.
    h_dimless = h*d/(G*M_initial); strain conventions match the original code.
    """
    a_star_0: float
    alpha_0: float
    max_n: int
    epsilon_seed: float
    LEVELS: List[Tuple[int, int, int]]
    ann_available: Dict[Tuple[int, int], bool]
    tr_pairs_idx: List[Tuple[int, int, int, int, int, int]]
    tau: np.ndarray
    ln_eps_all: np.ndarray
    eps_all: np.ndarray
    Mtil: np.ndarray
    astar: np.ndarray
    g_sr_all: np.ndarray
    ann_coeff_all: np.ndarray
    tr_coeff_matrix: np.ndarray
    h_dimless_all: np.ndarray
    h_dimless_tr: np.ndarray
    omega_ann_over_mu: np.ndarray
    omega_tr_over_mu: np.ndarray
    omega_ann_initial_over_mu: np.ndarray
    omega_tr_initial_over_mu: np.ndarray
    astar_f: float
    active_levels_t0: int
    dominant_level: Tuple[int, int, int]

    @property
    def alpha(self):
        return self.alpha_0 * self.Mtil

    @property
    def N_LEVELS(self):
        return len(self.LEVELS)


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
    h_all: np.ndarray           # (N_LEVELS, n_time)   physical h at r_kpc
    h_dimless_all: np.ndarray   # (N_LEVELS, n_time)   h * r / M0  (mass-independent)
    f_gw_hz: np.ndarray         # (N_LEVELS,)
    P_GW_total: np.ndarray      # (n_time,)

    # GW strain from transitions
    g_tr_matrix: np.ndarray     # (n_tr_pairs, n_time)
    h_tr_matrix: np.ndarray     # (n_tr_pairs, n_time)  physical h at r_kpc
    h_dimless_tr: np.ndarray    # (n_tr_pairs, n_time)  h_tr * r / M0
    f_tr_hz: np.ndarray         # (n_tr_pairs,)

    # Summary statistics
    astar_f: float
    t_final_yr: float
    epsilon_seed: float = 1e-60
    r_kpc: float = 1.0
    dimensionless: Optional[DimensionlessResults] = field(default=None, repr=False)
    active_levels_t0: int = 0
    dominant_level: Tuple[int, int, int] = field(default_factory=lambda: (2,1,1))


    @property
    def mu_eV(self):
        """Axion rest energy corresponding to alpha_0 and the chosen mass."""
        return self.MU * HBAR_EV_S / T_PL_S



def _positive_finite(name, value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def dimensionless_annihilation_rate(n, l, alpha_0, Mtil):
    """A = G*M_initial**2*Gamma_a/mu, evaluated without a physical mass.

    Evaluate the homogeneous rate formula at G=1, r_g=q, mu=alpha_0.
    Its numerical value is then A after dividing by alpha_0.
    """
    alpha = alpha_0 * Mtil
    omega = calc_omega_ann(r_g=Mtil, alpha=alpha, n=n)
    rate = calc_annihilation_rate(
        _level_string(n, l), alpha=alpha, omega=omega, G_N=1.0, r_g=Mtil)
    rate = float(rate) / alpha_0
    if not np.isfinite(rate) or rate < 0:
        raise ValueError(f"Invalid annihilation coefficient for {(n, l)}")
    return rate


def dimensionless_transition_rate(n_i, l_i, n_j, l_j, alpha_0, Mtil):
    """T = G*M_initial**2*Gamma_t/mu, with no physical mass conversion."""
    alpha = alpha_0 * Mtil
    omega = alpha_0 * alpha**2 / 2.0 * (1.0/n_j**2 - 1.0/n_i**2)
    key = f"{n_i}{_L_TO_SPEC[l_i]} {n_j}{_L_TO_SPEC[l_j]}"
    rate = calc_transition_rate(key, alpha, omega, G_N=1.0, r_g=Mtil)
    rate = float(rate) / alpha_0
    if not np.isfinite(rate) or rate < 0:
        raise ValueError(f"Invalid transition coefficient for {key}")
    return rate


def run_dimensionless(a_star_0=0.69, alpha_0=0.1, max_n=8,
                      epsilon_seed=1e-60, tau_end_factor=None, tau_end=None,
                      decay_decades=3.0):
    """Integrate once without specifying a PBH mass.

    epsilon_seed is the same initial normalized occupation in every level;
    it is NOT a one-particle seed at every physical mass. Growth delays and
    level competition depend on this choice. tau_end optionally supplies an
    explicit end time in tau=mu*t, which disables automatic continuation.
    Otherwise a growth window is followed by annihilation-tail integration
    until significant levels with annihilation channels (peak >= 1e-6 of
    the largest peak and >= 100 times the seed) fall decay_decades below
    their observed peaks. This does not bound growth of still-negligible
    modes at arbitrarily late times. Up to four tail
    extensions are used, with a warning if the target is not reached.
    Set decay_decades=0 to use only the growth window. tau_end_factor
    multiplies the initial window; explicit tau_end is also multiplied.
    The stiff system is solved with Radau and a supplied population Jacobian.
    Physical masses, years and strains are added later by to_physical().
    """
    automatic_end = tau_end is None
    if not np.isfinite(decay_decades) or decay_decades < 0:
        raise ValueError('decay_decades must be finite and nonnegative')
    _positive_finite('alpha_0', alpha_0)
    if not np.isfinite(a_star_0) or not 0 < a_star_0 < 1:
        raise ValueError("a_star_0 must lie strictly between 0 and 1")
    if not isinstance(max_n, (int, np.integer)) or not 2 <= max_n <= 8:
        raise ValueError("max_n must be an integer between 2 and 8")
    if not np.isfinite(epsilon_seed) or not 0 < epsilon_seed < 1:
        raise ValueError("epsilon_seed must lie strictly between 0 and 1")
    if tau_end_factor is not None:
        _positive_finite('tau_end_factor', tau_end_factor)
    if tau_end is not None:
        _positive_finite('tau_end', tau_end)

    levels = [(n, l, l) for l in range(1, max_n)
              for n in range(l+1, max_n+1)]
    count = len(levels)
    m_arr = np.array([m for n, l, m in levels])
    ann_available = {(n, l): _check_ann_available(n, l) for n, l, m in levels}
    lookup = {level: k for k, level in enumerate(levels)}
    pairs = [
        (lookup[(ni, li, mi)], lookup[(nj, lj, mj)], ni, li, nj, lj)
        for ni, li, mi in levels for nj, lj, mj in levels
        if ni > nj and _check_tr_available(ni, li, nj, lj)
    ]

    @lru_cache(maxsize=128)
    def ann_rates(q):
        return np.array([
            dimensionless_annihilation_rate(n, l, alpha_0, q)
            if ann_available[(n, l)] else 0.0 for n, l, m in levels])

    @lru_cache(maxsize=128)
    def tr_rates(q):
        return np.array([
            dimensionless_transition_rate(ni, li, nj, lj, alpha_0, q)
            for i, j, ni, li, nj, lj in pairs])

    def sr_rates(q, spin):
        return np.array([gamma_tilde_sr(alpha_0*q, spin, n, l, m)
                         for n, l, m in levels])

    # Each level carries positive angular momentum. Its occupation cannot
    # exceed the initial total angular momentum divided by m. This extension
    # of the RHS only bounds temporary solver guesses, not accepted states.
    ln_eps_cap = np.log((a_star_0 + epsilon_seed*np.sum(m_arr))/m_arr)

    def odes(tau, y):
        eps = np.exp(np.minimum(y[:count], ln_eps_cap))
        # This model only extracts BH mass; trial q>1 is unphysical.
        q = float(np.clip(y[count], 1e-9, 1.0))
        spin = float(np.clip(y[count+1], 0.0, 0.99999))
        sr = sr_rates(q, spin)
        dlogeps = sr - ann_rates(q)*eps
        for coefficient, (i, j, *_rest) in zip(tr_rates(q), pairs):
            dlogeps[i] -= coefficient*eps[j]
            dlogeps[j] += coefficient*eps[i]
        dq = -alpha_0*np.dot(sr, eps)
        dspin = np.dot(sr*eps, 2.0*alpha_0*q*spin-m_arr)/q**2
        derivative = np.r_[dlogeps, dq, dspin]
        if not np.all(np.isfinite(derivative)):
            raise FloatingPointError("Non-finite dimensionless ODE derivative")
        return derivative

    def jacobian(tau, y):
        # Use analytic occupation and SR derivatives. Numerical derivatives
        # across a quenched SR boundary create artificial feedback, while
        # perturbing empty levels can overflow SciPy's numerical Jacobian.
        eps = np.exp(np.minimum(y[:count], ln_eps_cap))
        deps = np.where(y[:count] < ln_eps_cap, eps, 0.0)
        q = float(np.clip(y[count], 1e-9, 1.0))
        spin = float(np.clip(y[count+1], 0.0, 0.99999))
        alpha = alpha_0*q
        sr = sr_rates(q, spin)
        sr_q = np.zeros(count)
        sr_spin = np.zeros(count)
        root = np.sqrt(1.0-spin**2)
        rp = 1.0+root
        rp_prime = -spin/root
        omega_h = spin/(2.0*rp)
        omega_prime = 1.0/(2.0*rp)-spin*rp_prime/(2.0*rp**2)
        for i, (n, l, m) in enumerate(levels):
            margin = m*omega_h-alpha
            if sr[i] <= 0 or margin <= 0:
                continue
            k = np.arange(1, l+1)
            z = spin*m-2.0*rp*alpha
            factors = k**2*(1.0-spin**2)+z**2
            log_g_alpha = np.sum(-4.0*rp*z/factors)
            log_g_spin = np.sum((-2.0*k**2*spin
                                 +2.0*z*(m-2.0*rp_prime*alpha))/factors)
            sr_q[i] = alpha_0*(sr[i]*((4*l+4)/alpha+log_g_alpha)
                                       -sr[i]/margin)
            sr_spin[i] = (sr[i]*(rp_prime/rp+log_g_spin)
                          +sr[i]/margin*m*omega_prime)

        jac = np.zeros((count+2, count+2))
        jac[np.arange(count), np.arange(count)] = -ann_rates(q)*deps
        for coefficient, (i, j, *_rest) in zip(tr_rates(q), pairs):
            jac[i, j] -= coefficient*deps[j]
            jac[j, i] += coefficient*deps[i]
        spin_factor = 2.0*alpha*spin-m_arr
        jac[count, :count] = -alpha_0*sr*deps
        jac[count+1, :count] = sr*deps*spin_factor/q**2

        # The remaining depletion-rate derivatives are smooth functions of q.
        step = min(1e-6, q/4.0)
        left, right = max(1e-9, q-step), min(1.0, q+step)
        ann_q = (ann_rates(right)-ann_rates(left))/(right-left)
        tr_q = (tr_rates(right)-tr_rates(left))/(right-left)
        jac[:count, count] = sr_q-ann_q*eps
        for coefficient, (i, j, *_rest) in zip(tr_q, pairs):
            jac[i, count] -= coefficient*eps[j]
            jac[j, count] += coefficient*eps[i]
        jac[:count, count+1] = sr_spin
        jac[count, count] = -alpha_0*np.dot(sr_q, eps)
        jac[count, count+1] = -alpha_0*np.dot(sr_spin, eps)
        spin_rhs = np.dot(sr*eps, spin_factor)/q**2
        jac[count+1, count] = (np.dot(sr_q*eps, spin_factor)/q**2
                              +2.0*alpha_0*spin*np.dot(sr, eps)/q**2
                              -2.0*spin_rhs/q)
        jac[count+1, count+1] = (np.dot(sr_spin*eps, spin_factor)/q**2
                                +2.0*alpha*np.dot(sr, eps)/q**2)
        if not 1e-9 <= y[count] <= 1.0:
            jac[:, count] = 0.0
        if not 0.0 <= y[count+1] <= 0.99999:
            jac[:, count+1] = 0.0
        return jac

    initial_rates = sr_rates(1.0, a_star_0)
    active = np.flatnonzero(initial_rates > 0)
    if not len(active):
        print(f"No initially superradiant levels at alpha={alpha_0:g}.")
        return None
    dom_idx = int(np.argmax(initial_rates))
    dominant_level = levels[dom_idx]
    g_dom = initial_rates[dom_idx]
    dom_m = dominant_level[2]
    astar_f = min(2.0*alpha_0*rhat_plus(a_star_0)/dom_m, a_star_0)

    if tau_end is None:
        # Estimate growth from the dimensionless seed rather than N=1.
        efolds = max(1.0, np.log(max(a_star_0, epsilon_seed)/epsilon_seed))
        efolds += 10*np.log(10.0)
        phase1 = efolds/g_dom
        phase2 = [efolds/g for (n, l, m) in levels if m > dom_m
                  for g in [gamma_tilde_sr(alpha_0, astar_f, n, l, m)]
                  if g > 0 and g >= g_dom/1e40]
        # Leave margin for rates to slow as earlier levels spin down the BH.
        tau_end = 10.0*(phase1 + max(phase2, default=0.0))
    if tau_end_factor is not None:
        tau_end *= tau_end_factor
    _positive_finite('integration end time', tau_end)

    print(f"Dimensionless BHSR: alpha_0={alpha_0:g}, a_star_0={a_star_0:g}, "
          f"n<={max_n}, epsilon_seed={epsilon_seed:g}")
    print(f"Tracking {count} levels, {len(pairs)} transitions; tau_end={tau_end:.4e}")
    y0 = [np.log(epsilon_seed)]*count + [1.0, a_star_0]
    # Growth, transitions and late annihilation span many orders of magnitude.
    # An implicit solver can reject large trial steps without overflowing the
    # rate formulae, and handles depleted modes without explicit-step stiffness.
    started = last_report = time.monotonic()

    def integration_progress(tau, y):
        # This constant, nonterminal event reports accepted steps only.
        # The timer affects logging, never the equations or stopping time.
        nonlocal last_report
        now = time.monotonic()
        if now-last_report >= 5.0:
            print(f"Integrating ({now-started:.0f}s elapsed): tau={tau:.4e}, "
                  f"M/M_initial={y[count]:.6f}, spin={y[count+1]:.6f}",
                  flush=True)
            last_report = now
        return 1.0

    print("Integrating with Radau; progress is reported about every 5 seconds. "
          "Large multilevel runs can take a few minutes.", flush=True)
    def integrate_interval(start, end, initial):
        solution = solve_ivp(
            odes, (start, end), initial, method='Radau', jac=jacobian,
            rtol=1e-9, atol=1e-12, dense_output=False,
            events=integration_progress)
        if not solution.success:
            raise RuntimeError(f"Dimensionless integration failed: {solution.message}")
        if not np.all(np.isfinite(solution.y)):
            raise RuntimeError("Integration returned non-finite accepted states")
        q_values, spin_values = solution.y[count:]
        tolerance = 1e-8
        if (np.any(q_values <= 0) or np.any(q_values > 1.0+tolerance)
                or np.any(spin_values < -tolerance)
                or np.any(spin_values > a_star_0+tolerance)
                or np.any(solution.y[:count] > ln_eps_cap[:, None]+tolerance)):
            raise RuntimeError("Integration returned an accepted state outside the physical bounds")
        return solution

    sol = integrate_interval(0.0, tau_end, y0)
    time_segments = [sol.t]
    state_segments = [sol.y]
    peak_logs = sol.y[:count].max(axis=1)

    if automatic_end and decay_decades > 0:
        # After the growth stage, estimate annihilation tails from the actual
        # populations and mass. For d eps/d tau=-A eps**2, the time to fall
        # from eps_now to eps_target is (1/eps_target-1/eps_now)/A.
        # Ignore seed-level/negligible populations so they cannot set an
        # arbitrarily long horizon. All levels remain in the coupled ODE.
        for extension in range(5):
            final = sol.y[:, -1]
            target_logs = peak_logs-decay_decades*np.log(10.0)
            significant = peak_logs >= max(
                np.log(epsilon_seed)+np.log(100.0),
                peak_logs.max()+np.log(1e-6))
            coefficients = ann_rates(float(final[count]))
            pending = significant & (coefficients > 0) & (final[:count] > target_logs)
            if not np.any(pending):
                break
            if extension == 4:
                import warnings
                warnings.warn(
                    "The automatic window ended before all significant levels "
                    "reached the requested decay. Set tau_end explicitly to "
                    "inspect a longer interval.", RuntimeWarning, stacklevel=2)
                break
            remaining = ((np.exp(-target_logs[pending])
                          - np.exp(-final[:count][pending]))/coefficients[pending])
            next_end = max(2.0*sol.t[-1], sol.t[-1]+1.25*float(remaining.max()))
            _positive_finite('automatic decay end time', next_end)
            print(f"Extending through annihilation decay: tau={sol.t[-1]:.4e} "
                  f"to {next_end:.4e} "
                  f"(target: {decay_decades:g} decades below peak).", flush=True)
            sol = integrate_interval(sol.t[-1], next_end, final)
            time_segments.append(sol.t[1:])
            state_segments.append(sol.y[:, 1:])
            peak_logs = np.maximum(peak_logs, sol.y[:count].max(axis=1))

    tau = np.concatenate(time_segments)
    state = np.concatenate(state_segments, axis=1)
    q, spin = state[count:]
    eps = np.exp(state[:count])
    print(f"ODE solver reached tau={tau[-1]:.4e}; computing strain outputs ...",
          flush=True)
    alpha = alpha_0*q
    sr = np.empty((count, len(tau)))
    ann = np.empty_like(sr)
    tr = np.empty((len(pairs), len(tau)))
    last_report = time.monotonic()
    for column, (qi, ai) in enumerate(zip(q, spin)):
        sr[:, column] = sr_rates(qi, ai)
        ann[:, column] = ann_rates(qi)
        tr[:, column] = tr_rates(qi)
        now = time.monotonic()
        if now-last_report >= 5.0:
            print(f"Preparing strains: {column+1}/{len(tau)} time samples "
                  f"({now-started:.0f}s elapsed)", flush=True)
            last_report = now
    w_ann = np.array([2.0*(1.0-alpha**2/(2.0*n**2)) for n, l, m in levels])
    w_ann_initial = np.array([2.0*(1.0-alpha_0**2/(2.0*n**2))
                              for n, l, m in levels])
    if np.any(w_ann <= 0):
        raise ValueError("Non-positive annihilation frequency in hydrogenic model")
    h_ann = eps*np.sqrt(8.0*ann/w_ann)
    w_tr = np.empty((len(pairs), len(tau)))
    h_tr = np.empty_like(w_tr)
    w_tr_initial = np.empty(len(pairs))
    for p, (i, j, ni, li, nj, lj) in enumerate(pairs):
        delta_n = 1.0/nj**2-1.0/ni**2
        w_tr[p] = alpha**2/2.0*delta_n
        w_tr_initial[p] = alpha_0**2/2.0*delta_n
        # Preserve the existing simulation's transition-strain prefactor.
        h_tr[p] = np.sqrt(eps[i])*np.sqrt(eps[j])*np.sqrt(8.0*tr[p]/w_tr[p])

    print(f"Integration complete: {len(tau)} samples; "
          f"M/M_initial={q[-1]:.6f}, a_star={spin[-1]:.6f}")
    return DimensionlessResults(
        a_star_0=a_star_0, alpha_0=alpha_0, max_n=max_n,
        epsilon_seed=epsilon_seed, LEVELS=levels, ann_available=ann_available,
        tr_pairs_idx=pairs, tau=tau, ln_eps_all=state[:count], eps_all=eps,
        Mtil=q, astar=spin, g_sr_all=sr, ann_coeff_all=ann,
        tr_coeff_matrix=tr, h_dimless_all=h_ann, h_dimless_tr=h_tr,
        omega_ann_over_mu=w_ann, omega_tr_over_mu=w_tr,
        omega_ann_initial_over_mu=w_ann_initial,
        omega_tr_initial_over_mu=w_tr_initial, astar_f=astar_f,
        active_levels_t0=len(active), dominant_level=dominant_level)


def to_physical(results: DimensionlessResults, M_BH_solar: float,
                r_kpc: float = 1.0):
    """Convert a saved/in-memory solution without integrating again.

    Fixed alpha_0 implies mu scales inversely with the chosen initial mass.
    N scales as mass**2, time as mass, strain as mass/distance, and frequency
    as 1/mass. Frequencies in peak tables use initial alpha, as before.
    """
    if not isinstance(results, DimensionlessResults):
        raise TypeError("to_physical expects a DimensionlessResults object")
    _positive_finite('M_BH_solar', M_BH_solar)
    _positive_finite('r_kpc', r_kpc)
    M0 = M_BH_solar*M_SUN_PLANCK
    norm = M0**2
    mu = results.alpha_0/M0
    time_unit = M0*T_PL_S/results.alpha_0
    N = norm*results.eps_all
    M_BH = M0*results.Mtil
    m_arr = np.array([m for n, l, m in results.LEVELS])
    J_cloud = m_arr @ N
    J_BH = results.astar*M_BH**2
    J_total = J_BH+J_cloud
    M_irr = np.array([irreducible_mass(m, a)
                      for m, a in zip(M_BH, results.astar)])
    g_ann = results.ann_coeff_all/norm
    return SimulationResults(
        M_BH_solar=M_BH_solar, a_star_0=results.a_star_0,
        alpha_0=results.alpha_0, max_n=results.max_n, M0=M0, MU=mu,
        TAU_TO_YR=time_unit/YEAR_S, LEVELS=results.LEVELS,
        _L_ARR=np.array([l for n, l, m in results.LEVELS]), _M_ARR=m_arr,
        N_LEVELS=results.N_LEVELS, ann_available=results.ann_available,
        tr_pairs_idx=results.tr_pairs_idx, tau=results.tau,
        lnN_all=results.ln_eps_all+np.log(norm), NORM=norm,
        eps_all=results.eps_all, Mtil=results.Mtil, astar=results.astar,
        t_yr=results.tau*time_unit/YEAR_S, N_all=N, alpha=results.alpha,
        M_BH=M_BH, J_cloud=J_cloud, M_cloud=mu*N.sum(axis=0),
        J_BH=J_BH, J_total=J_total,
        dJ_frac=np.abs((J_total-J_total[0])/J_total[0]), M_irr=M_irr,
        E_rot=M_BH-M_irr, dM_irr=M_irr-M_irr[0],
        g_sr_all=results.g_sr_all, g_ann_all=g_ann,
        h_all=results.h_dimless_all*M0/(r_kpc*KPC_PLANCK),
        h_dimless_all=results.h_dimless_all,
        f_gw_hz=results.omega_ann_initial_over_mu*mu*OMEGA_PL_TO_HZ,
        P_GW_total=(2.0*g_ann*N**2).sum(axis=0),
        g_tr_matrix=results.tr_coeff_matrix/norm,
        h_tr_matrix=results.h_dimless_tr*M0/(r_kpc*KPC_PLANCK),
        h_dimless_tr=results.h_dimless_tr,
        f_tr_hz=results.omega_tr_initial_over_mu*mu*OMEGA_PL_TO_HZ,
        astar_f=results.astar_f, t_final_yr=results.tau[-1]*time_unit/YEAR_S,
        active_levels_t0=results.active_levels_t0,
        dominant_level=results.dominant_level, epsilon_seed=results.epsilon_seed,
        r_kpc=r_kpc, dimensionless=results)


def run_simulation(M_BH_solar=None, a_star_0=0.65, alpha_0=0.6,
                   max_n=6, tau_end_factor=None, epsilon_seed=1e-60,
                   tau_end=None, decay_decades=3.0):
    """Run without a mass, or optionally convert to the old physical result.

    Omit M_BH_solar to receive DimensionlessResults. Supplying it retains the
    previous plotting/peak-table interface; the integration is identical.
    """
    if M_BH_solar is not None:
        _positive_finite('M_BH_solar', M_BH_solar)
    results = run_dimensionless(a_star_0=a_star_0, alpha_0=alpha_0,
                                max_n=max_n, epsilon_seed=epsilon_seed,
                                tau_end_factor=tau_end_factor, tau_end=tau_end,
                                decay_decades=decay_decades)
    if results is None or M_BH_solar is None:
        return results
    return to_physical(results, M_BH_solar)


def _dimensionless_filename(results):
    # Round-trip precision avoids overwriting nearby points in a dense grid.
    return (f"dimless_alpha{float(results.alpha_0)!r}_a{float(results.a_star_0)!r}"
            f"_n{results.max_n}_seed{float(results.epsilon_seed)!r}.npz")


def save_dimensionless(results, filepath=None):
    """Save a mass-free NPZ (schema 2), including seed and frequency ratios.

    Accepts either DimensionlessResults or a converted SimulationResults.
    No reference PBH mass or physical frequency is needed in the archive.
    """
    if isinstance(results, SimulationResults):
        results = results.dimensionless
    if not isinstance(results, DimensionlessResults):
        raise TypeError("save_dimensionless expects a dimensionless solution")
    if filepath is None:
        filepath = Path(_THIS_DIR)/'Data'/_dimensionless_filename(results)
    filepath = Path(filepath)
    if filepath.suffix != '.npz':
        filepath = Path(str(filepath)+'.npz')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(vars(results))
    arrays['schema_version'] = 2
    arrays['alpha'] = results.alpha
    arrays['LEVELS'] = np.asarray(results.LEVELS, dtype=int)
    arrays['ann_available'] = np.array([
        results.ann_available[(n, l)] for n, l, m in results.LEVELS], dtype=bool)
    arrays['tr_pairs_idx'] = np.asarray(results.tr_pairs_idx, dtype=int).reshape(-1, 6)
    arrays['level_labels'] = np.array([f'{n}{l}{m}' for n, l, m in results.LEVELS])
    arrays['tr_labels'] = np.array([
        f'{ni}{li}{li}->{nj}{lj}{lj}' for i, j, ni, li, nj, lj in results.tr_pairs_idx],
        dtype=str)
    np.savez_compressed(filepath, **arrays)
    guide = filepath.with_name(filepath.stem+'_rescaling.txt')
    guide.write_text(
        'Mass-independent BHSR solution (schema 2)\n'
        f'alpha_0={results.alpha_0!r}, a_star_0={results.a_star_0!r}, '
        f'max_n={results.max_n}, epsilon_seed={results.epsilon_seed!r}\n'
        'epsilon_seed is a fixed dimensionless seed, not N_initial=1.\n'
        'tau=mu*t; q=M/M_initial; epsilon=N/(G*M_initial**2).\n'
        'Load with load_dimensionless(path), then to_physical(solution, M_solar).\n'
        'Or use rescale_to_mass(path, M_solar, r_kpc=1.0) for a dictionary.\n'
        'At fixed alpha_0, spin and seed: N scales as M**2, t as M, '
        'h as M/d, frequency and axion mass as 1/M.\n', encoding='utf-8')
    print(f'Dimensionless results saved to {filepath}')
    return str(filepath)


def load_dimensionless(filepath):
    """Load schema-2 results. Old one-particle-seed files are not universal."""
    with np.load(filepath, allow_pickle=False) as data:
        if 'schema_version' not in data or int(data['schema_version']) != 2:
            raise ValueError('This is a legacy archive with a mass-dependent seed. '
                             'Regenerate it with run_dimensionless(), or use '
                             'rescale_to_mass() for legacy approximate rescaling.')
        fields = {name: data[name].copy()
                  for name in DimensionlessResults.__dataclass_fields__}
    for name in ('a_star_0', 'alpha_0', 'epsilon_seed', 'astar_f'):
        fields[name] = float(fields[name])
    for name in ('max_n', 'active_levels_t0'):
        fields[name] = int(fields[name])
    fields['LEVELS'] = [tuple(map(int, row)) for row in fields['LEVELS']]
    fields['ann_available'] = {
        (n, l): bool(available) for (n, l, m), available
        in zip(fields['LEVELS'], fields['ann_available'])}
    fields['tr_pairs_idx'] = [tuple(map(int, row)) for row in fields['tr_pairs_idx']]
    fields['dominant_level'] = tuple(map(int, fields['dominant_level']))
    return DimensionlessResults(**fields)


def rescale_to_mass(filepath, M_new_solar, r_kpc=1.0):
    """Return physical arrays at any mass/distance, with no ODE integration.

    Supports old archives with an explicit warning about their seed; new
    schema-2 files are exactly rescalable at fixed dimensionless inputs.
    Use to_physical(load_dimensionless(path), mass) for plotting/peak tables.
    """
    _positive_finite('M_new_solar', M_new_solar)
    _positive_finite('r_kpc', r_kpc)
    with np.load(filepath, allow_pickle=False) as data:
        is_new = 'schema_version' in data
        if not is_new:
            import warnings
            warnings.warn('Legacy archive: rescaling preserves its reference '
                          'dimensionless seed, not N_initial=1 at the new mass.',
                          UserWarning, stacklevel=2)
            alpha_0 = float(data['alpha_0'])
            M0 = M_new_solar*M_SUN_PLANCK
            mu = alpha_0/M0
            return {
                **{key: data[key].copy() for key in
                   ('tau', 'eps_all', 'Mtil', 'astar', 'alpha', 'level_labels', 'tr_labels')},
                't_yr': data['tau']*M_new_solar*GM_SUN_OVER_C3/(alpha_0*YEAR_S),
                'N_all': M0**2*data['eps_all'],
                'h_ann': data['h_dimless_all']*M0/(r_kpc*KPC_PLANCK),
                'h_tr': data['h_dimless_tr']*M0/(r_kpc*KPC_PLANCK),
                'f_gw_hz': data['f_gw_hz']*float(data['M0_solar'])/M_new_solar,
                'f_tr_hz': data['f_tr_hz']*float(data['M0_solar'])/M_new_solar,
                'M_planck': M0, 'alpha_0': alpha_0,
                'a_star_0': float(data['a_star_0']), 'r_kpc': r_kpc,
                'mu_eV': mu*HBAR_EV_S/T_PL_S,
            }
    dim = load_dimensionless(filepath)
    physical = to_physical(dim, M_new_solar, r_kpc)
    return {
        'tau': dim.tau, 'eps_all': dim.eps_all, 'Mtil': dim.Mtil,
        'astar': dim.astar, 'alpha': dim.alpha, 't_yr': physical.t_yr,
        'N_all': physical.N_all, 'h_ann': physical.h_all,
        'h_tr': physical.h_tr_matrix, 'f_gw_hz': physical.f_gw_hz,
        'f_tr_hz': physical.f_tr_hz, 'M_planck': physical.M0,
        'M_BH_solar': M_new_solar, 'alpha_0': dim.alpha_0,
        'a_star_0': dim.a_star_0, 'epsilon_seed': dim.epsilon_seed,
        'r_kpc': r_kpc, 'mu_eV': physical.MU*HBAR_EV_S/T_PL_S,
        'level_labels': np.array([f'{n}{l}{m}' for n, l, m in dim.LEVELS]),
        'tr_labels': np.array([f'{ni}{li}{li}->{nj}{lj}{lj}'
                               for i, j, ni, li, nj, lj in dim.tr_pairs_idx]),
    }


def compute_peak_characteristics(results: SimulationResults):
    """
    Compute for each annihilation level and transition pair:
    - peak strain h_max (at results.r_kpc)
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
        print("  Annihilation Peak Characteristics (sorted by decreasing peak strain)")
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
        print("  Transition Peak Characteristics (sorted by decreasing peak strain)")
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
    if results.r_kpc != 1.0:
        raise ValueError("Detector peak tables require strains at 1 kpc; "
                         "convert with to_physical(dim, mass, r_kpc=1.0).")

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


def plot_occupations_and_spin(results, save_path=None, units="dimensionless"):
    """Plot spin and log10 occupation, dimensionless by default.

    units="dimensionless" accepts DimensionlessResults or SimulationResults
    and displays tau=mu*t and epsilon=N/(G*M_initial**2).
    units="physical" requires a result converted with to_physical(), and
    displays years and N. No mass conversion is performed by this function.
    """
    if results is None:
        print("No simulation results to plot.")
        return

    if units not in ("dimensionless", "physical"):
        raise ValueError('units must be "dimensionless" or "physical"')
    if isinstance(results, DimensionlessResults):
        log_eps = results.ln_eps_all/np.log(10)
    elif isinstance(results, SimulationResults):
        log_eps = (results.lnN_all-np.log(results.NORM))/np.log(10)
    else:
        raise TypeError("Expected DimensionlessResults or SimulationResults")
    if units == "dimensionless":
        plot_time = results.tau
        log_population = log_eps
        xlabel = r"$\tau = \mu_a t$"
        ylabel = r"$\log_{10}\,\epsilon_i$"
        floor = np.log10(results.epsilon_seed)
    else:
        if not isinstance(results, SimulationResults):
            raise ValueError("Physical axes require to_physical(solution, M_BH_solar)")
        plot_time = results.t_yr
        log_population = results.lnN_all/np.log(10)
        xlabel = r"$t$ [yr]"
        ylabel = r"$\log_{10}\,N_i$"
        floor = float(log_population[:, 0].min())
    astar = results.astar
    LEVELS = results.LEVELS

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

    color_spin = "k"

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

    fig_main = plt.figure(figsize=(5, 4))
    gs_main  = GridSpec(2, 1, figure=fig_main,
                        height_ratios=[0.5, 3.5], hspace=0.1)
    ax_spin  = fig_main.add_subplot(gs_main[0])
    ax_main  = fig_main.add_subplot(gs_main[1], sharex=ax_spin)

    # ── Spin panel ────────────────────────────────────────────────────
    ax_spin.plot(plot_time, astar, color='k', lw=1.2)
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
        xy=(plot_time[plot_time > 0][0] if np.any(plot_time > 0) else plot_time[1], results.a_star_0),
        xytext=(6, -10), textcoords="offset points",
        fontsize=8, color=color_spin,
    )
    ax_spin.set_xscale("log")

    # ── Occupation panel ──────────────────────────────────────────────
    active_labels = []   # (latex_str, colour, peak_log_population) for inline labels

    for k, (n, l, m) in enumerate(LEVELS):
        col, ls, lw = level_style_map[(n, l, m)]
        is_active = log_eps[k].max() > np.log10(results.epsilon_seed)+0.5
        ax_main.plot(
            plot_time, log_population[k],
            color=col, ls=ls, lw=lw,
            alpha=1.0 if is_active else 0.25,
        )
        if is_active:
            active_labels.append(
                (rf"$|{n}{l}{m}\rangle$", col, float(log_population[k].max()))
            )

    ax_main.set_xlabel(xlabel, fontsize=12)
    ax_main.set_ylabel(ylabel, fontsize=12)
    ax_main.grid(True, alpha=0.25, linestyle="--")
    top = float(log_population.max())
    padding = max(1.0, 0.04*(top-floor))
    ax_main.set_ylim(floor-padding, top+padding)
    t_min_pos = plot_time[plot_time > 0][0] if np.any(plot_time > 0) else plot_time[1]
    ax_main.set_xlim(t_min_pos, plot_time[-1])
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
    ax_tr_h.set_ylabel(fr"$h_{{\rm tr}}$ (at {results.r_kpc:g}\,kpc)", fontsize=12)
    ax_tr_h.grid(True, alpha=0.25, which="both", linestyle="--")
    ax_tr_h.legend(fontsize=8, loc="upper left", ncol=1)
    ax_tr_h.set_xscale('log')
    t_min_pos = t_yr[t_yr > 0][0] if np.any(t_yr > 0) else t_yr[1]
    ax_tr_h.set_xlim(t_min_pos, t_yr[-1])

    fig_tr.suptitle(
        fr"GW strain from axion transitions — $m=l$ levels ($n\leq{results.max_n}$), "
        fr"$M_{{\rm BH}} = {results.M_BH_solar}\,M_\odot$, "
        fr"$\alpha_0 = {results.alpha_0}$, distance $= {results.r_kpc:g}\,\rm kpc$",
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

def run_alpha_sweep(M_BH_solar=None, a_star_0=0.69, max_n=8,
                    alpha_min=0.001, alpha_max=1.5, n_alpha_points=10,
                    output_dir=None, epsilon_seed=1e-60,
                    tau_end_factor=None, tau_end=None, decay_decades=3.0):
    """Save mass-independent solutions on the existing linear alpha grid.

    Specify only spin, coupling grid, max_n and epsilon_seed for reusable NPZ
    files. Optionally supply M_BH_solar to also produce the existing physical
    peak tables for GWDetector_alpha.py. Convert any saved solution to other
    masses later with to_physical()/rescale_to_mass(), without reintegration.

    Returns one DimensionlessResults per grid point (None if SR inactive),
    or SimulationResults when an output mass is explicitly requested.
    """
    _positive_finite('alpha_min', alpha_min)
    _positive_finite('alpha_max', alpha_max)
    if alpha_max < alpha_min:
        raise ValueError('alpha_max must be at least alpha_min')
    if not isinstance(n_alpha_points, (int, np.integer)) or n_alpha_points < 1:
        raise ValueError('n_alpha_points must be a positive integer')
    if M_BH_solar is not None:
        _positive_finite('M_BH_solar', M_BH_solar)
    if output_dir is None:
        output_dir = Path(_THIS_DIR)/'Data'/'alpha_sweep'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_grid = np.linspace(alpha_min, alpha_max, n_alpha_points)
    print(f'Alpha sweep: {n_alpha_points} linearly spaced points, '
          f'a_star_0={a_star_0:g}, n<={max_n}, epsilon_seed={epsilon_seed:g}')
    all_results = []
    for i, alpha in enumerate(alpha_grid):
        print(f'\nSweep point {i+1}/{n_alpha_points}: alpha={alpha:.6g}')
        dim = run_dimensionless(a_star_0=a_star_0, alpha_0=float(alpha),
                                max_n=max_n, epsilon_seed=epsilon_seed,
                                tau_end_factor=tau_end_factor, tau_end=tau_end,
                                decay_decades=decay_decades)
        if dim is None:
            all_results.append(None)
            continue
        save_dimensionless(dim, output_dir/_dimensionless_filename(dim))
        if M_BH_solar is None:
            all_results.append(dim)
        else:
            physical = to_physical(dim, M_BH_solar)
            ann_char, tr_char = compute_peak_characteristics(physical)
            print_peak_tables(ann_char, tr_char)
            save_peak_tables(ann_char, tr_char, physical, output_dir=output_dir)
            all_results.append(physical)
    print(f'Alpha sweep complete: {sum(r is not None for r in all_results)}'
          f'/{n_alpha_points} superradiant points saved.')
    return all_results


def main():
    # Dimensionless integration inputs: no PBH mass is used here.
    A_STAR_0 = 0.69
    ALPHA_0 = 0.10                 # single-run coupling
    MAX_N = 3
    EPSILON_SEED = 1e-60           # fixed epsilon, NOT a one-particle seed
    TAU_END_FACTOR = None          # optional multiplier of the initial window
    TAU_END = 1e50                 # explicit tau endpoint; disables automatic tails
    DECAY_DECADES = 3.0            # follow significant levels below their peaks

    RUN_ALPHA_SWEEP = False       # True: save reusable solutions across the grid
    N_ALPHA_POINTS = 40
    ALPHA_MIN = 0.01
    ALPHA_MAX = 0.8

    # Optional physical peak tables. The main plot always uses tau and epsilon.
    # Leave M_BH_SOLAR=None for an entirely dimensionless run.
    M_BH_SOLAR = None
    DISTANCE_KPC = 1.0

    if RUN_ALPHA_SWEEP:
        run_alpha_sweep(a_star_0=A_STAR_0, max_n=MAX_N,
                        alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX,
                        n_alpha_points=N_ALPHA_POINTS, epsilon_seed=EPSILON_SEED,
                        tau_end_factor=TAU_END_FACTOR, tau_end=TAU_END,
                        decay_decades=DECAY_DECADES)
        return

    dim = run_dimensionless(a_star_0=A_STAR_0, alpha_0=ALPHA_0,
                            max_n=MAX_N, epsilon_seed=EPSILON_SEED,
                            tau_end_factor=TAU_END_FACTOR, tau_end=TAU_END,
                            decay_decades=DECAY_DECADES)
    if dim is None:
        return
    save_dimensionless(dim)
    plot_dir = os.path.join(_THIS_DIR, 'Plots')
    plot_occupations_and_spin(
        dim, units="dimensionless",
        save_path=os.path.join(plot_dir, 'superradiance_main.pdf'))

    if M_BH_SOLAR is not None:
        results = to_physical(dim, M_BH_SOLAR, r_kpc=DISTANCE_KPC)
        ann_char, tr_char = compute_peak_characteristics(results)
        print_peak_tables(ann_char, tr_char)
        # Detector peak tables retain their 1 kpc normalization.
        tables = results if DISTANCE_KPC == 1.0 else to_physical(dim, M_BH_SOLAR)
        ann_tables, tr_tables = compute_peak_characteristics(tables)
        save_peak_tables(ann_tables, tr_tables, tables,
                         output_dir=os.path.join(_THIS_DIR, 'Data'))
    else:
        print("For physical strain tables, set M_BH_SOLAR or convert a saved "
              "solution with to_physical().")
    plt.show()
    plt.close('all')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nRun cancelled by Ctrl+C.", flush=True)
        sys.exit(130)
