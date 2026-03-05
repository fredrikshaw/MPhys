import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


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

# Function to parse Mathematica-style numbers
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

# Custom file reader for Mathematica-style data
def load_file(filepath):
    def parse_row(row):
        return [parse_mathematica_number(x) for x in row]
    
    with open(filepath, 'r') as f:
        data = [parse_row(line.split()) for line in f]
    return pd.DataFrame(data, columns=["alpha", "CF_Gamma", "Other_Column"])

# Function to parse quantum numbers and spin state from the filename
def parse_quantum_numbers_and_spin(filepath):
    fname = Path(filepath).name
    quantum_match = re.search(r'n(\d+)l(\d+)m(\d+)', fname)
    spin_match = re.search(r'at(\d+\.?\d*)', fname)  # Match spin state in the form "at0.99"
    if quantum_match and spin_match:
        n, l, m = map(int, quantum_match.groups())
        a_t = float(spin_match.group(1))
        # Format a_t to up to 3 decimal places, removing trailing zeros
        a_t_str = format(a_t, '.3f').rstrip('0').rstrip('.')
        spin_label = rf"$\tilde{{a}} = {a_t_str}$"  # Format spin state as LaTeX
        return rf"$|{n}{l}{m}\rangle$", [n, l, m], a_t, spin_label
    return fname, None, None, None

# Files and colors for plotting
FILES = [
    "2. Relativistic Superradiance Rate/Mathematica/SR_n2l1m1_at0.69_aMin0.02_aMax0.25.dat",
    "2. Relativistic Superradiance Rate/Mathematica/SR_n2l1m1_at0.70_aMin0.02_aMax0.25.dat",
    "2. Relativistic Superradiance Rate/Mathematica/SR_n2l1m1_at0.80_aMin0.05_aMax0.30.dat",
    "2. Relativistic Superradiance Rate/Mathematica/SR_n2l1m1_at0.90_aMin0.05_aMax0.50.dat",
    "2. Relativistic Superradiance Rate/Mathematica/SR_n2l1m1_at0.99_aMin0.05_aMax0.50.dat",
    "2. Relativistic Superradiance Rate/Mathematica/SR_n2l1m1_at0.990_aMin0.03_aMax0.52.dat",
]
COLOURS = ["blue", "green", "orange", "red", "purple", "black"]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}"
})

fig, ax = plt.subplots(figsize=(7, 5.5))

for filepath, colour in zip(FILES, COLOURS):
    label, quantum_numbers, a_t, spin_label = parse_quantum_numbers_and_spin(filepath)
    df = load_file(filepath)

    if quantum_numbers and a_t is not None:
        n, l, m = quantum_numbers
        r_g = 1  # Example gravitational radius [eV^-1]

        # Filter out rows with invalid or non-positive CF_Gamma values
        df_cf = df[df["CF_Gamma"] > 0].dropna(subset=["CF_Gamma"])
        if not df_cf.empty:
            ax.plot(df_cf["alpha"], df_cf["CF_Gamma"], color=colour, linestyle="solid", linewidth=1.5, label=spin_label)

# ── Legend ────────────────────────────────────────────────────────────────────
ax.legend(fontsize=10, frameon=False)

ax.set_yscale("log")
ax.set_xlabel(r"$\alpha$", fontsize=13)
ax.set_ylabel(r"$\Gamma^{\mathrm{sr}} r_g$", fontsize=13)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.set_ylim(1e-13, 1e-6)
ax.set_xlim(0, 0.52)

plt.tight_layout()

output_path = Path("2. Relativistic Superradiance Rate/Plots/superradiance_plot.pdf")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150)
plt.show()