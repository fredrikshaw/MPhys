import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re

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
FILE = "2. Relativistic Superradiance Rate/Mathematica/Convergence/SR_convergence_n2l1m1_at0.500_20260402.csv"

# ── Colours (reuse style) ────────────────────────────────────────────────────
COLOURS = ["#e03c3c", "#e07c3c", "#7842f5", "#284945", "#284245"]

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

fig, ax = plt.subplots(figsize=(4, 3.5))

# ── Load data ────────────────────────────────────────────────────────────────
df = load_file(FILE)

# Expect columns: alpha, N_max, SR_rate
# (adjust if names differ slightly)
alpha_values = sorted(df["alpha"].unique())

# ── Plot each alpha ──────────────────────────────────────────────────────────
for i, alpha in enumerate(alpha_values):
    colour = COLOURS[i % len(COLOURS)]

    df_alpha = df[df["alpha"] == alpha].sort_values("Nmax")

    ax.plot(
        df_alpha["Nmax"],
        df_alpha["SR_rate"],
        color=colour,
        linewidth=1.5,
        label=rf"$\alpha = {alpha:.3f}$"
    )

# ── Formatting ───────────────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$N_{\mathrm{max}}$", fontsize=13)
ax.set_ylabel(r"$\Gamma^{\mathrm{sr}}r_g$", fontsize=13)

ax.grid(True, which="both", linestyle="--", alpha=0.4)

ax.legend(fontsize=9, frameon=False)

plt.tight_layout()

# ── Save ─────────────────────────────────────────────────────────────────────
output_path = "2. Relativistic Superradiance Rate/Plots/sr_convergence_plot.pdf"

plt.savefig(output_path, dpi=150)
plt.show()