import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── LaTeX / CM Roman styling ───────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
})

# ── derived bound on α/ℓ ──────────────────────────────────────────────────
def alpha_over_ell_max(a_star):
    """Upper bound on α/ℓ from the superradiance condition (m = ℓ)."""
    return a_star / (2.0 * (1.0 + np.sqrt(1.0 - a_star**2)))

# ── spin grid (avoid a*=1 exactly for numerical safety) ───────────────────
a_vals = np.linspace(0.0, 0.9999, 2000)
aol_max = alpha_over_ell_max(a_vals)

# ── figures ────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(6.5, 5.5))
fig1.patch.set_facecolor("white")

fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
fig2.patch.set_facecolor("white")

# ── palette ───────────────────────────────────────────────────────────────
SR_COLOR   = "#4C72B0"   # blue fill for SR region
BOUND_COLOR = "#C44E52"  # red curve
ANNOT_COLOR = "#2ca02c"  # green annotations

# ══════════════════════════════════════════════════════════════════════════
# LEFT PANEL  –  α/ℓ vs a*
# ══════════════════════════════════════════════════════════════════════════
ax = ax1

ax.fill_between(a_vals, 0, aol_max,
        color=SR_COLOR, alpha=0.25)
ax.plot(a_vals, aol_max,
        color=BOUND_COLOR, lw=2.2, label=r"$(\alpha/\ell)_{\max}$")

ax.text(0.8, 0.1, "Superradiant Region",
    fontsize=14, color="black", weight="bold",
    ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.6))

# reference lines
ax.axhline(0.5, color="k", ls="--", lw=1.2, alpha=0.7)
ax.axvline(1.0, color="k", ls=":",  lw=1.0, alpha=0.5)

# annotation: extremal limit
ax.annotate(r"Extremal limit: $(\alpha/\ell)_{\max} \to \frac{1}{2}$",
            xy=(0.98, 0.495), xytext=(0.55, 0.42),
            arrowprops=dict(arrowstyle="->", color=ANNOT_COLOR, lw=1.4),
            fontsize=12, color=ANNOT_COLOR,
            ha="center")

# sample spin markers
for a_s in [0.9, 0.7, 0.5]:
    aol_s = alpha_over_ell_max(a_s)
    ax.plot(a_s, aol_s, "o", color=BOUND_COLOR, ms=6, zorder=5)
    ax.annotate(f"$\tilde a={a_s}$\n"
                r"$(\alpha/\ell)_{\rm max}$" + f"$={aol_s:.3f}$",
                xy=(a_s, aol_s), xytext=(a_s - 0.22, aol_s + 0.03),
                fontsize=12, color="k",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

ax.set_xlabel(r" $\tilde a$", fontsize=15)
ax.set_ylabel(r"$\alpha / \ell$", fontsize=15)

ax.set_xlim(0, 1.0)
ax.set_ylim(0, 0.5)
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_minor_locator(MultipleLocator(0.025))
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.grid(True, which="major", ls="--", alpha=0.35)

# ══════════════════════════════════════════════════════════════════════════
# RIGHT PANEL  –  r_c / r_g along the SR boundary  (= ℓ²/α²)
# ══════════════════════════════════════════════════════════════════════════
# On the SR boundary α/ℓ = f(a*), so ℓ/α = 1/f  ⟹  r_c/r_g = n²(ℓ/α)²
# use n = ℓ+1 → n/ℓ → 1 for large ℓ; show the n=ℓ+1 case explicitly

n_over_ell = np.array([2, 1.5, 1.2, 1.0 + 1e-6])   # n/ℓ  (ℓ=1,2,5,∞)
labels      = [r"$n=2\ell$", r"$n=3\ell/2$", r"$n=6\ell/5$", r"$n=\ell$ (min)"]
colors      = ["#1f77b4","#ff7f0e","#2ca02c","#9467bd"]

# avoid a* = 0 (divergence)
a_plot = np.linspace(0.01, 0.9999, 2000)
f_vals = alpha_over_ell_max(a_plot)          # = (α/ℓ)_max on boundary

for nol, lab, col in zip(n_over_ell, labels, colors):
    rc_over_rg = nol**2 / f_vals**2          # r_c/r_g = (n/α)² = (n/ℓ)²·(ℓ/α)²
    ax2.semilogy(a_plot, rc_over_rg, color=col, lw=2, label=lab)

ax2.axhline(1, color="k", ls="--", lw=1.2, alpha=0.6)
ax2.text(0.02, 1.3, r"$r_c = r_g$ (horizon scale)", fontsize=9, alpha=0.7)

ax2.fill_between(a_plot,
                 1,
                 (1.0 + 1e-6)**2 / f_vals**2,
                 color=SR_COLOR, alpha=0.10)

ax2.set_xlabel(r"$\tilde a$", fontsize=13)
ax2.set_ylabel(r"$r_c / r_g$  (cloud radius / gravitational radius)", fontsize=11)

ax2.set_xlim(0, 1.02)
ax2.set_ylim(0.8, 2e4)
ax2.tick_params(which="both", direction="in", top=True, right=True)
ax2.legend(fontsize=10, loc="upper right")
ax2.grid(True, which="both", ls="--", alpha=0.3)

# inset note
ax2.text(0.50, 0.12,
         r"Even at extremal spin ($\tilde a\to1$), $r_c/r_g\geq 4$" "\n"
         r"$\Rightarrow$ hydrogenic approximation always valid",
         transform=ax2.transAxes, fontsize=9,
         ha="center", va="bottom",
         bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="goldenrod", alpha=0.9))

fig1.tight_layout()
fig2.tight_layout()

fig1.savefig("5. General Report Plots/superradiance_region_ax1.pdf",
             dpi=200, bbox_inches="tight")

fig2.savefig("5. General Report Plots/superradiance_region_ax2.pdf",
             dpi=200, bbox_inches="tight")

print("Saved separate ax1 and ax2 PDF/PNG plots.")
plt.show()

# ── quick numerical check ─────────────────────────────────────────────────
print("\nNumerical check of (α/ℓ)_max at selected spins:")
for a in [0.0, 0.5, 0.9, 0.99, 1.0 - 1e-9]:
    print(f"  a* = {a:.4f}  →  (α/ℓ)_max = {alpha_over_ell_max(a):.6f}")