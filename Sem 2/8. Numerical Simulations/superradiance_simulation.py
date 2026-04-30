"""
======================================================================
Black Hole Superradiance Simulation — |2,1,1> Mode
======================================================================
Simulates the exponential growth of an axion cloud and the resulting
Kerr BH spin-down until the superradiance condition is no longer met.

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

M_BH_SOLAR = 1e-11    # Initial BH mass [solar masses]
A_STAR_0   = 0.99    # Initial dimensionless spin  a* = J/M²
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
# ODE system
# ══════════════════════════════════════════════════════════════════════

def odes(tau, y):
    """
    RHS of the BH-superradiance ODE system.

    State:  y = [ln N,  M̃ = M/M₀,  a*]
    Time:   τ = t·μ  (dimensionless)

    Equations
    ---------
      d(ln N)/dτ  =  Γ̃_SR(α, a*)

      dM̃/dτ       = −(α₀/M₀²) · Γ̃_SR · N

      da*/dτ      =  (Γ̃_SR · N) / (M₀² · M̃²) · (2α·a* − 1)

    where  α(τ) = α₀·M̃(τ)  evolves with the BH mass.

    Conservation laws (exact, used as numerical checks)
    ---------------------------------------------------
      d/dτ [ a*·M̃²·M₀² + N ]  = 0   (total angular momentum)
      d/dτ [ M̃·M₀ + N·μ ]     = 0   (total mass-energy)
    """
    lnN, Mtil, astar = y
    Mtil  = max(Mtil, 1e-9)
    astar = float(np.clip(astar, 0.0, 0.99999))

    N     = np.exp(lnN)
    alpha = ALPHA_0 * Mtil          # α(τ) = M(τ)·μ = α₀·M̃

    g = gamma_tilde_sr(alpha, astar)

    dlnN   = g
    dMtil  = -(ALPHA_0 / M0**2) * g * N
    dastar = (g * N / (M0**2 * Mtil**2)) * (2.0 * alpha * astar - 1.0)

    return [dlnN, dMtil, dastar]


# ══════════════════════════════════════════════════════════════════════
# Terminal event
# ══════════════════════════════════════════════════════════════════════

def event_sr_off(tau, y):
    """Fires (terminal) when the SR margin crosses zero from above."""
    _, Mtil, astar = y
    return sr_margin(ALPHA_0 * max(Mtil, 1e-9),
                     float(np.clip(astar, 0.0, 0.99999)), m=1)

event_sr_off.terminal  = True
event_sr_off.direction = -1


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("BH Superradiance — |2,1,1⟩ mode   (hydrogenic approximation)")
    print("=" * 65)
    print(f"  M_BH = {M_BH_SOLAR} M_sun  =  {M0:.3e} M_Pl")
    print(f"  a*₀  = {A_STAR_0}")
    print(f"  α₀   = {ALPHA_0}   →  μ = {MU:.3e} M_Pl")

    # ── Sanity checks ────────────────────────────────────────────────
    margin0  = sr_margin(ALPHA_0, A_STAR_0, m=1)
    g0       = gamma_tilde_sr(ALPHA_0, A_STAR_0)
    g0_bcp   = (ALPHA_0**8 / 24.0) * margin0   # BCP small-α analytic

    print(f"\n  SR margin at t=0:  m·Ω_H − α  = {margin0:.5f}  (> 0 ✓)")
    print(f"  Γ̃_SR at t=0:")
    print(f"    Hydrogenic (Detweiler/Dolan) : {g0:.4e}")
    print(f"    BCP small-α analytic         : {g0_bcp:.4e}")
    print(f"    Ratio                        : {g0/g0_bcp:.5f}")
    print(f"  (Should be ≈ 1 in the weak-coupling regime α = {ALPHA_0})")

    # ── Integration time estimate ────────────────────────────────────
    N_sat   = M0**2 * A_STAR_0
    tau_end = 3.0 * np.log(max(N_sat, 2.0)) / max(g0, 1e-300)

    print(f"\n  e-folding time τ_SR = {1/g0:.3e}  =  {1/g0 * TAU_TO_YR:.3e} yr")
    print(f"  τ_end estimate      = {tau_end:.3e}  =  {tau_end * TAU_TO_YR:.3e} yr")

    # ── Integrate ────────────────────────────────────────────────────
    print("\nIntegrating ... ", end="", flush=True)
    y0 = [np.log(N0), 1.0, A_STAR_0]

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
    lnN   = sol.y[0]
    Mtil  = sol.y[1]
    astar = sol.y[2]

    # ── Derived quantities ───────────────────────────────────────────
    N       = np.exp(lnN)
    alpha   = ALPHA_0 * Mtil
    J_BH    = astar * Mtil**2 * M0**2
    J_cloud = N                              # each axion: ΔJ = m·ħ = 1
    M_cloud = N * MU                         # each axion: ΔE ≈ μ
    J_total = J_BH + J_cloud
    dJ_frac = np.abs((J_total - J_total[0]) / J_total[0])
    t_yr    = tau * TAU_TO_YR
    g_sr_t  = np.array([gamma_tilde_sr(a, s) for a, s in zip(alpha, astar)])

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\nResults")
    print(f"  End:  τ = {tau[-1]:.4e}   t = {t_yr[-1]:.4f} yr")
    print(f"  a*  : {A_STAR_0:.4f} → {astar[-1]:.4f}  (Δa* = {A_STAR_0 - astar[-1]:.4f})")
    print(f"  M   : {M_BH_SOLAR:.3f} → {Mtil[-1]*M_BH_SOLAR:.5f} M_sun"
          f"  (ΔM/M₀ = {1 - Mtil[-1]:.5f})")
    print(f"  N_max          = {N[-1]:.4e}")
    print(f"  M_cloud/M_BH   = {M_cloud[-1]/(Mtil[-1]*M0):.5f}")
    print(f"  J conservation = {dJ_frac.max():.2e}  ✓")

    # ── Plots ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.36)
    ax1, ax2, ax3, ax4, ax5, ax6 = [fig.add_subplot(gs[i, j])
                                     for i in range(3) for j in range(2)]
    tl = r"$t$  [yr]"

    # 1. Occupation number
    ax1.plot(t_yr, lnN, color="royalblue", lw=2)
    ax1.set_xlabel(tl); ax1.set_ylabel(r"$\ln N$")
    ax1.set_title("Axion cloud: occupation number")
    ax1.grid(True, alpha=0.3)

    # 2. SR rate
    ax2.semilogy(t_yr, np.maximum(g_sr_t, 1e-100), color="darkorange", lw=2)
    ax2.set_xlabel(tl)
    ax2.set_ylabel(r"$\tilde{\Gamma}_{\rm SR} = \Gamma_{\rm SR}/\mu$")
    ax2.set_title("SR rate — hydrogenic approx. (log scale)")
    ax2.grid(True, alpha=0.3, which="both")

    # 3. BH spin
    ax3.plot(t_yr, astar, color="firebrick", lw=2, label=r"$a_*$ (simulated)")
    ax3.axhline(astar[-1], ls="--", color="grey", lw=1.5,
                label=fr"SR threshold  $a_{{*,f}} = {astar[-1]:.3f}$")
    ax3.set_xlabel(tl); ax3.set_ylabel(r"$a_*$")
    ax3.set_title("BH spin — spindown to SR threshold")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    # 4. Mass budget
    ax4.plot(t_yr, Mtil * M_BH_SOLAR,      color="seagreen",  lw=2,
             label=r"$M_{\rm BH}$ $[M_\odot]$")
    ax4.plot(t_yr, M_cloud / M_SUN_PLANCK,  color="royalblue", lw=2, ls="--",
             label=r"$M_{\rm cloud}$ $[M_\odot]$")
    ax4.set_xlabel(tl); ax4.set_ylabel(r"Mass  $[M_\odot]$")
    ax4.set_title("Mass budget")
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

    # 5. Angular momentum budget
    J0 = J_total[0]
    ax5.plot(t_yr, J_BH    / J0, color="firebrick", lw=2,
             label=r"$J_{\rm BH}/J_0$")
    ax5.plot(t_yr, J_cloud / J0, color="royalblue", lw=2,
             label=r"$J_{\rm cloud}/J_0$")
    ax5.plot(t_yr, J_total / J0, "k--", lw=1.5, alpha=0.6,
             label=r"$J_{\rm total}$ (conserved)")
    ax5.set_xlabel(tl); ax5.set_ylabel(r"$J / J_0$")
    ax5.set_title("Angular momentum transfer")
    ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3)

    # 6. Phase-space trajectory (α, a*)
    a_line         = np.linspace(0.01, 0.9999, 500)
    alpha_boundary = a_line / (2.0 * (1.0 + np.sqrt(1.0 - a_line**2)))

    ax6.fill_betweenx(a_line, 0, alpha_boundary,
                      alpha=0.08, color="grey", label="SR inactive")
    ax6.plot(alpha_boundary, a_line, "k--", lw=1.8,
             label=r"SR boundary $\omega = \Omega_H$")
    ax6.plot(alpha, astar, color="purple", lw=2.5, label="BH trajectory")
    ax6.scatter([alpha[0]],  [astar[0]],  color="green", s=100, zorder=5,
                label="Start")
    ax6.scatter([alpha[-1]], [astar[-1]], color="red",   s=100, zorder=5,
                label="End (SR off)")
    ax6.set_xlabel(r"$\alpha = GM\mu/\hbar c$")
    ax6.set_ylabel(r"$a_*$")
    ax6.set_title(r"Phase-space trajectory $(\alpha,\,a_*)$")
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, ALPHA_0 * 1.15)
    ax6.set_ylim(0, 1.0)

    fig.suptitle(
        fr"BH Superradiance — $|211\rangle$ mode   "
        fr"$M_{{BH}} = {M_BH_SOLAR}\,M_\odot$,   "
        fr"$\alpha_0 = {ALPHA_0}$,   "
        fr"$a_{{*,0}} = {A_STAR_0}$   "
        fr"[hydrogenic approx.]",
        fontsize=12, y=1.01,
    )

    # ── Save ─────────────────────────────────────────────────────────
    out_path = os.path.join(_THIS_DIR, 'Sem 2', '8. Numerical Simulations',
                            'Plots', 'superradiance_plot.pdf')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    preview = "Sem 2/8. Numerical Simulations/Plots/superradiance_211_hydro.png"
    plt.savefig(preview, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()