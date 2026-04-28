import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import re
import numpy as np


# Add the directory containing the ParamCalculator script to the Python path.
# We compute it relative to this file so the script works on different machines.
current_dir = Path(__file__).resolve().parent
sem2_dir = None
for p in current_dir.parents:
    if p.name == "Sem 2":
        sem2_dir = p
        break
if sem2_dir is None:
    sem2_dir = current_dir.parent
script_dir = sem2_dir / "0. Scripts from Sem 1"
if not script_dir.exists():
    # fallback: search upward for the folder name
    for p in current_dir.parents:
        candidate = p / "0. Scripts from Sem 1"
        if candidate.exists():
            script_dir = candidate
            break
sys.path.append(str(script_dir.resolve()))

# Import the script (expects `calc_superradiance_rate` in ParamCalculator)
from ParamCalculator import calc_superradiance_rate


# ── Configuration ────────────────────────────────────────────────────────────
FILES = [
    "Sem 2/2. Relativistic Superradiance Rate/Mathematica/SR_n3l2m2_at0.990_aMin0.010_aMax2.082_20260428.dat",
]


COLOURS = ["#e03c3c", "#e07c3c", "#7842f5", "#284945"]
# ─────────────────────────────────────────────────────────────────────────────


def parse_mathematica_number(s):
    if isinstance(s, float):
        return s
    s = str(s).strip().strip('"')
    s = re.sub(r'`[\d.]+\.\*\^', 'e', s)
    s = re.sub(r'`[\d.]+\.?$', '', s)
    s = re.sub(r'\*\^', 'e', s)
    try:
        return float(s)
    except ValueError:
        return None


def parse_quantum_numbers(filepath):
    fname = Path(filepath).name
    match = re.search(r'n(\d+)l(\d+)m(\d+)', fname)
    if match:
        n, l, m = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return rf"$|{n}{l}{m}\rangle$", [n, l, m]
    return fname, None


def load_file(filepath):
    df = pd.read_csv(filepath, sep="\t")
    df.columns = df.columns.str.strip().str.strip('"')
    df["alpha"]       = df["alpha"].apply(parse_mathematica_number)
    df["CF_Gamma"]    = df["CF_Gamma"].apply(parse_mathematica_number)
    df["Hydro_Gamma"] = df["Hydro_Gamma"].apply(parse_mathematica_number)
    df = df.dropna(subset=["alpha"])
    return df


# ── Plot setup ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}"
})

fig, ax = plt.subplots(figsize=(4, 3.5))

# track max x for setting x-limits later
max_x = 0.0
min_x = float('inf')

for filepath, colour in zip(FILES, COLOURS):
    label, quantum_numbers = parse_quantum_numbers(filepath)
    df = load_file(filepath)

    if quantum_numbers:
        n, l, m = quantum_numbers
        a_star = 0.99

        # CF data: positive values only
        df_cf = df[df["CF_Gamma"] > 0].dropna(subset=["CF_Gamma"])
        if not df_cf.empty:
            # compute hydrogenic (non-relativistic) rate at the same alpha values
            df_cf = df_cf.copy()
            df_cf["Hydro_Gamma_Calc"] = df["Hydro_Gamma"] #df_cf["alpha"].apply(
            #    lambda alpha: calc_superradiance_rate(l, m, n, a_star, 1, alpha)
            #)

            # keep only positive hydro calcs to avoid divide-by-zero / sign flips
            df_cf = df_cf[df_cf["Hydro_Gamma_Calc"] > 0]
            if not df_cf.empty:
                df_cf["ratio_minus_one"] = (df_cf["CF_Gamma"] / df_cf["Hydro_Gamma_Calc"]) 
                # x-axis is alpha divided by l
                xvals = df_cf["alpha"] / float(l)
                ax.plot(xvals, df_cf["ratio_minus_one"],
                    color=colour, linestyle="solid", linewidth=1.5, label=label)
                if len(xvals):
                    max_x = max(max_x, float(xvals.max()))
                    min_x = min(min_x, float(xvals.min()))

        # annotate near the maximum of the CF curve if available
        if not df[df["CF_Gamma"] > 0].empty:
            df_cf_all = df[df["CF_Gamma"] > 0].dropna(subset=["CF_Gamma"]).copy()
            peak_idx = df_cf_all["CF_Gamma"].idxmax()
            peak_alpha = df_cf_all.loc[peak_idx, "alpha"] / float(l)
            # place label slightly above zero line for readability
            #ax.text(peak_alpha, 0.05, label,
            #    color=colour, fontsize=11, ha="center", va="bottom")


# ── Legend & styling ────────────────────────────────────────────────────────
ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.8)
ax.legend(fontsize=10, frameon=False)

# Use a symmetric log scale on the y-axis to compress large outliers while
# keeping the region near zero linear for readability.
ax.set_yscale('log')
ax.set_xscale("log")

# x-axis label: alpha divided by l
ax.set_xlabel(r"$\alpha / \ell$", fontsize=13)
ax.set_ylabel(r"$(\Im(\omega_{CF}/\omega_{\mathrm{hy}}))$", fontsize=13)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.set_xlim(0, 0.5)

plt.tight_layout()

output_path = Path("2. Relativistic Superradiance Rate/Plots/cf_hydro_ratio_plot.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150)
plt.show()
