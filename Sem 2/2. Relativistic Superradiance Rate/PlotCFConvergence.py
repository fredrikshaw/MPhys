import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# ── Path handling (same structure as your script) ─────────────────────────────
current_dir = Path(__file__).resolve().parent
sem2_dir = None
for p in current_dir.parents:
    if p.name == "Sem 2":
        sem2_dir = p
        break
if sem2_dir is None:
    sem2_dir = current_dir.parent

# File (adjust name if needed)
FILE = "Sem 2/2. Relativistic Superradiance Rate/Mathematica/Convergence/SR_convergence_n2l1m1_at0.500_20260424.csv"


# ── Colours (reuse style) ────────────────────────────────────────────────────
COLOURS = ["#2B7BB9", "#38A09C", "#6DAA65", "#D4A832", "#C85D38"]

# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_mathematica_number(s):
    if isinstance(s, float) or isinstance(s, int):
        return float(s)
    s = str(s).strip().strip('"')
    s = re.sub(r'`[\d.]+\.\*\^', 'e', s)
    s = re.sub(r'`[\d.]+\.?$', '', s)
    s = re.sub(r'\*\^', 'e', s)
    try:
        return float(s)
    except ValueError:
        return None


def load_file(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.strip('"')

    # Apply parsing
    for col in df.columns:
        df[col] = df[col].apply(parse_mathematica_number)

    return df.dropna()


# ── Plot setup (identical style) ──────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}"
})

fig, ax = plt.subplots(figsize=(4, 2.5))

# ── Load data ────────────────────────────────────────────────────────────────
df = load_file(FILE)

# Expect columns: alpha, N_max, SR_rate
# (adjust if names differ slightly)
alpha_values = sorted(df["alpha"].unique())

# ── Plot each alpha ──────────────────────────────────────────────────────────
for i, alpha in enumerate(alpha_values):
    colour = COLOURS[i % len(COLOURS)]

    df_alpha = df[df["alpha"] == alpha].sort_values("Nmax")

    y = df_alpha["SR_rate"]
    y_final = y.iloc[-1]

    rel_err = np.abs(y / y_final - 1)

    # Drop the final point (it's identically zero by construction)
    # and any other points that hit floating-point zero
    mask = rel_err > 0

    ax.plot(
        df_alpha["Nmax"][mask],
        rel_err[mask],
        color=colour,
        linewidth=1.5,
        marker="o",
        markersize=3.5,
        label=rf"{alpha:.2f}",
        zorder=1
    )



# ── Formatting ───────────────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$N$", fontsize=13)
ax.set_ylabel(r"$|\Im(\tilde\omega_N)/\Im(\tilde\omega_{\rm fin}) - 1|$", fontsize=13)

ax.grid(True, which="both", linestyle="--", alpha=0.4)


# ── Convergence threshold ────────────────────────────────────────────────
ERROR_LEVEL = 1e-3   # 0.1% = 1e-3 (change this freely)

# Horizontal error threshold line
ax.axhline(
    ERROR_LEVEL,
    color="black",
    linestyle="--",
    linewidth=1
)
ax.text(
    ax.get_xlim()[0]*1.1,
    ERROR_LEVEL * 1.1,
    rf"${100*ERROR_LEVEL:.2g}\%$",
    fontsize=9,
    verticalalignment="bottom",
    zorder=2
)

ax.legend(fontsize=9, frameon=False, loc="lower left", title=r"$\alpha$")

plt.tight_layout()

# ── Save ─────────────────────────────────────────────────────────────────────
output_path = "Sem 2/2. Relativistic Superradiance Rate/Plots/sr_convergence_plot.pdf"

plt.savefig(output_path, dpi=150)
plt.show()