import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── LaTeX formatting consistent with report ───────────────────────────────────
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.linewidth": 0.8,
})

# ── Kerr parameters (geometrised units: G = c = 1) ────────────────────────────
M       = 1.0
a_tilde = 0.99
a       = a_tilde * M

r_outer = M + np.sqrt(M**2 - a**2)
r_inner = M - np.sqrt(M**2 - a**2)

theta = np.linspace(0, 2 * np.pi, 2000)
r_ergo = M + np.sqrt(np.clip(M**2 - a**2 * np.cos(theta)**2, 0, None))
x_ergo = r_ergo * np.sin(theta)
z_ergo = r_ergo * np.cos(theta)

phi = np.linspace(0, 2 * np.pi, 2000)
x_out = r_outer * np.cos(phi);  z_out = r_outer * np.sin(phi)
x_inn = r_inner * np.cos(phi);  z_inn = r_inner * np.sin(phi)

# ── Colour palette ────────────────────────────────────────────────────────────
col_ergo_fill = "#dce8f5"
col_ergo_edge = "#2166ac"
col_out_fill  = "#d4d4d4"
col_out_edge  = "#1a1a2e"
col_inn_fill  = "#f5f5f5"
col_inn_edge  = "#c0392b"
col_sing      = "#1a1a2e"
col_ann       = "#222222"
col_spin      = "#555555"

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 3.5), facecolor="white")
ax.set_facecolor("white")

# Filled regions
ax.fill(x_ergo, z_ergo, color=col_ergo_fill, zorder=1)
ax.fill(x_out,  z_out,  color=col_out_fill,  zorder=2)
ax.fill(x_inn,  z_inn,  color=col_inn_fill,  zorder=3)

# Boundary curves
lw = 1.4
ax.plot(x_ergo, z_ergo, color=col_ergo_edge, lw=lw,        zorder=6,
        label=r"Ergosphere, $r_{\mathrm{ergo}}(\theta)$")
ax.plot(x_out,  z_out,  color=col_out_edge,  lw=lw + 0.4,  zorder=7,
        label=r"Event horizon, $r_+$")
ax.plot(x_inn,  z_inn,  color=col_inn_edge,  lw=lw, ls="--", dashes=(5,3), zorder=8,
        label=r"Cauchy horizon, $r_-$")

# Ring singularity
ax.plot(0, 0, "o", ms=4.5, color=col_sing, zorder=10,
        label=r"Ring singularity")

# Spin axis
ax_len = 2.55 * M
ax.annotate("", xy=(0, ax_len), xytext=(0, -ax_len),
            arrowprops=dict(arrowstyle="<->", color=col_spin,
                            lw=0.75, mutation_scale=8), zorder=5)
ax.text(0.07, ax_len * 0.90, r"$\hat{z}$", color=col_spin, fontsize=9, va="center")

# Annotations
"""ann_kw = dict(fontsize=8.5, color=col_ann, va="center",
              arrowprops=dict(arrowstyle="-|>", color=col_ann,
                              lw=0.65, mutation_scale=7,
                              connectionstyle="arc3,rad=0.0"),
              zorder=11)"""

idx_eq = np.argmin(np.abs(theta - np.pi / 2))
"""ax.annotate(r"$r_{\mathrm{ergo}}(\pi/2) = %.4f\,M$" % r_ergo[idx_eq],
            xy=(r_ergo[idx_eq], 0.0), xytext=(2.05, 1.25), ha="left", **ann_kw)"""

"""ang = np.pi / 5
ax.annotate(r"$r_+ = %.4f\,M$" % r_outer,
            xy=(r_outer * np.sin(ang), r_outer * np.cos(ang)),
            xytext=(1.65, 1.90), ha="left", **ann_kw)

ax.annotate(r"$r_- = %.4f\,M$" % r_inner,
            xy=(-r_inner * np.sin(np.pi/3), r_inner * np.cos(np.pi/3)),
            xytext=(-2.65, 0.95), ha="left", **ann_kw)

ax.annotate(r"Ring singularity",
            xy=(0.0, 0.0), xytext=(-2.4, -0.60), ha="left", **ann_kw)"""

# Axes
ax.set_xlim(-2.5 * M, 3 * M)
ax.set_ylim(-2 * M, 2 * M)
ax.set_aspect("equal")
ax.set_xlabel(r"$x\,/\,M$", fontsize=11)
ax.set_ylabel(r"$z\,/\,M$", fontsize=11)
ax.tick_params(labelsize=9, direction="in", top=True, right=True, length=4, width=0.7)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color("#333333")

# Legend
leg = ax.legend(loc="lower right", fontsize=8, frameon=True,
                framealpha=1.0, facecolor="white", edgecolor="#aaaaaa",
                borderpad=0.7, handlelength=2.5, labelspacing=0.55)
leg.get_frame().set_linewidth(0.6)

plt.tight_layout(pad=0.5)
plt.savefig("5. General Report Plots/kerr_black_hole.pdf",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

print("Saved.")