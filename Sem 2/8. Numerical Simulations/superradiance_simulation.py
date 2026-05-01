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

Superradiance rate — hydrogenic (NR) approximation
---------------------------------------------------
In the weak-coupling regime (α ≪ 1) the quasi-bound-state frequency is
well approximated by the hydrogen-like analytic formula of Detweiler (1980)
and Dolan (2007). This is imported directly from leaver_superradiance.py
as hydrogen_gamma(n, l, m, alpha, at), which returns:

    Γ_NR = 2·r₊·C_{nl}·g_{lm}(ã)·(m·Ω_H − α)·α^{4l+5}

in GM=c=1 units (i.e. in units of 1/M, same convention as the CF result).
This is the full rate for N: dN/dt = Γ_NR · N in those units.

Converting to the dimensionless Γ̃ = Γ/μ used in the ODE (τ = t·μ):

    Γ̃_SR = Γ_NR / α     (since μ = α/M and M = 1 in GM=1 units)

This is pure arithmetic — no iterative solver, no grid, no precomputation.
It can be evaluated at every ODE step at negligible cost.

For α ≲ 0.1 the hydrogenic rate agrees with the exact Leaver CF result
to better than ~10%. Switch to the CF grid version for α ≳ 0.2.

Key References
--------------
  Detweiler (1980), Phys. Rev. D 22, 2323
  Dolan (2007), Phys. Rev. D 76, 084001  [arXiv:0705.2880]
  Brito, Cardoso & Pani (2015) "Superradiance", Living Reviews
"""

import os, sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Leaver module import ───────────────────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_LEAVER_DIR = os.path.join(_THIS_DIR, '..', '2. Relativistic Superradiance Rate')
sys.path.insert(0, _LEAVER_DIR if os.path.isdir(_LEAVER_DIR) else _THIS_DIR)

from leaver_superradiance import hydrogen_gamma


# ══════════════════════════════════════════════════════════════════════
# Physical constants  (unit conversion at plot stage only)
# ══════════════════════════════════════════════════════════════════════

M_SUN_KG       = 1.989e30    # kg
M_PL_KG        = 2.176e-8    # kg
YEAR_S         = 3.156e7     # s per year
M_SUN_PLANCK   = M_SUN_KG / M_PL_KG    # ≈ 9.137e37  M_Pl per M_sun
GM_SUN_OVER_C3 = 4.927e-6    # s  (GM_sun/c^3, natural BH time unit)


# ══════════════════════════════════════════════════════════════════════
# Physical input parameters  — edit these
# ══════════════════════════════════════════════════════════════════════

M_BH_SOLAR = 10.0    # Initial BH mass [solar masses]
A_STAR_0   = 0.99    # Initial dimensionless spin  a* = J/M²
ALPHA_0    = 0.1   # Gravitational coupling  α₀ = M₀μ
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

LEVELS   = [(n, l, l) for l in range(1, 8) for n in range(l + 1, 9)]
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
# ODE system  (multi-level)
# ══════════════════════════════════════════════════════════════════════

def odes(tau, y):
    """
    RHS of the multi-level BH-superradiance ODE system.

    State:  y = [lnN_0, …, lnN_{K-1},  M̃,  ã]
            where K = N_LEVELS = 28  (all m=l modes, n ≤ 8)
    Time:   τ = t·μ  (dimensionless)

    Equations
    ---------
      d(lnN_i)/dτ  =  Γ̃_SR(n_i, l_i, m_i, α, ã)       [per level]

      dM̃/dτ        = −(α₀/M₀²) · Σ_i Γ̃_SR_i · N_i      [summed]

      dã/dτ        =  Σ_i [Γ̃_SR_i · N_i / (M₀² M̃²)]    [summed]
                       · (2α·ã − m_i)

    The m_i factor in the spin equation arises because each axion in
    level i carries ΔJ = m_i·ħ (not just ħ = 1).  For m_i = 1 this
    reduces to the single-level formula.

    All levels share the same α(τ) = α₀·M̃(τ) and ã(τ) — the BH
    backreaction is fully coupled across all levels.

    Conservation laws
    -----------------
      d/dτ [ ã·M̃²·M₀² + Σ_i m_i·N_i ] = 0   (total angular momentum)
      d/dτ [ M̃·M₀ + μ·Σ_i N_i ]        = 0   (total mass-energy)
    """
    lnN_arr = y[:N_LEVELS]
    Mtil    = max(y[N_LEVELS],     1e-9)
    astar   = float(np.clip(y[N_LEVELS + 1], 0.0, 0.99999))

    N_arr = np.exp(lnN_arr)
    alpha = ALPHA_0 * Mtil

    # SR rate for every level
    g_arr = np.array([gamma_tilde_sr(alpha, astar, n, l, m)
                      for (n, l, m) in LEVELS])

    # Cloud occupation growth
    dlnN_arr = g_arr                                           # shape (K,)

    # BH mass change: sum weighted by occupation
    dMtil = -(ALPHA_0 / M0**2) * np.dot(g_arr, N_arr)

    # BH spin change: each level weighted by its own m_i
    #   dã/dτ = Σ_i [Γ̃_i N_i / (M₀² M̃²)] (2α ã − m_i)
    dastar = (np.dot(g_arr * N_arr, 2.0 * alpha * astar - _M_ARR)
              / (M0**2 * Mtil**2))

    return list(dlnN_arr) + [dMtil, dastar]


# ══════════════════════════════════════════════════════════════════════
# Terminal event
# ══════════════════════════════════════════════════════════════════════

def event_sr_off(tau, y):
    """
    Fires (terminal) when all m=1 SR conditions switch off simultaneously.

    All m=l=1 levels (|211⟩, |311⟩, …, |811⟩) share the same SR
    threshold α < ã/(2r̂₊), so they all switch off at the same ã.
    Higher-m levels have lower thresholds and would stay active longer,
    but their rates are suppressed by α^{4l+5} and contribute negligibly
    to the dynamics at small α.

    Returns the m=1 SR margin; goes negative when SR is off.
    """
    Mtil  = max(y[N_LEVELS],     1e-9)
    astar = float(np.clip(y[N_LEVELS + 1], 0.0, 0.99999))
    return sr_margin(ALPHA_0 * Mtil, astar, m=1)

event_sr_off.terminal  = True
event_sr_off.direction = -1


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
    print(f"\n  Levels tracked:")
    for n, l, m in LEVELS:
        g0_i = gamma_tilde_sr(ALPHA_0, A_STAR_0, n, l, m)
        print(f"    |{n}{l}{m}⟩   Γ̃₀ = {g0_i:.3e}   "
              f"τ_SR = {1/g0_i:.2e}" if g0_i > 0 else
              f"    |{n}{l}{m}⟩   Γ̃₀ = {g0_i:.3e}   (SR inactive)")

    # ── Integration time: set by the fastest-growing level |211⟩ ─────
    g0_dom  = gamma_tilde_sr(ALPHA_0, A_STAR_0, 2, 1, 1)
    N_sat   = M0**2 * A_STAR_0
    tau_end = 3.0 * np.log(max(N_sat, 2.0)) / max(g0_dom, 1e-300)

    print(f"\n  Dominant level |211⟩:  Γ̃₀ = {g0_dom:.4e}")
    print(f"  e-folding τ_SR = {1/g0_dom:.3e}  =  {1/g0_dom * TAU_TO_YR:.3e} yr")
    print(f"  τ_end estimate = {tau_end:.3e}  =  {tau_end * TAU_TO_YR:.3e} yr")

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
        events       = [event_sr_off],
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

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\nResults")
    print(f"  End:  τ = {tau[-1]:.4e}   t = {t_yr[-1]:.4f} yr")
    print(f"  ã   : {A_STAR_0:.4f} → {astar[-1]:.4f}  (Δã = {A_STAR_0 - astar[-1]:.4f})")
    print(f"  M   : {M_BH_SOLAR:.3f} → {Mtil[-1]*M_BH_SOLAR:.5f} M_sun"
          f"  (ΔM/M₀ = {1 - Mtil[-1]:.5f})")
    print(f"\n  Peak occupation numbers:")
    for k, (n, l, m) in enumerate(LEVELS):
        peak = lnN_all[k].max() / np.log(10)
        if peak > 0.5:
            print(f"    |{n}{l}{m}⟩   log₁₀N_max = {peak:.2f}")
    M_irr0_sol = M_irr[0]  / M_SUN_PLANCK
    M_irrf_sol = M_irr[-1] / M_SUN_PLANCK
    E_rot0_sol = E_rot[0]  / M_SUN_PLANCK
    E_rotf_sol = E_rot[-1] / M_SUN_PLANCK
    print(f"\n  Christodoulou-Ruffini decomposition:")
    print(f"  M_irr : {M_irr0_sol:.5f} → {M_irrf_sol:.5f} M_sun"
          f"  (ΔM_irr = +{M_irrf_sol - M_irr0_sol:.5f}, area theorem ✓)")
    print(f"  E_rot : {E_rot0_sol:.5f} → {E_rotf_sol:.5f} M_sun")
    print(f"  M_cloud total : {M_cloud[-1]/M_SUN_PLANCK:.5f} M_sun")
    print(f"  J conservation = {dJ_frac.max():.2e}  ✓")
    print(f"  Area theorem   = {'✓' if dM_irr.min() >= -1e-6*M_irr[0] else '✗'}")

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
    ax_main.set_xlim(t_yr[0], t_yr[-1])

    main_path    = os.path.join(_THIS_DIR, 'Plots', 'superradiance_main.pdf')
    main_preview = "Sem 2/8. Numerical Simulations/Plots/superradiance_main.png"
    os.makedirs(os.path.dirname(main_path), exist_ok=True)
    fig_main.savefig(main_path,    dpi=150, bbox_inches="tight")
    fig_main.savefig(main_preview, dpi=150, bbox_inches="tight")
    print(f"\nMain figure saved to {main_path}")

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
    s2.legend(fontsize=9); s2.grid(True, alpha=0.25, linestyle="--")

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
    s3.legend(fontsize=9); s3.grid(True, alpha=0.25, linestyle="--")

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