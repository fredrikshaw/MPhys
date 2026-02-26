from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import re

# ── Configuration ────────────────────────────────────────────────────────────
FILES = [
    "Mathematica/SR_n2l1m1_at0.99_aMin0.05_aMax0.50.dat"
]

COLOURS = ["#e03c3c", "#e07c3c", "#7842f5", "#2a9d8f"]
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
        n, l, m = match.group(1), match.group(2), match.group(3)
        return rf"$|{n}{l}{m}\rangle$"
    return fname


def load_file(filepath):
    df = pd.read_csv(filepath, sep="\t")
    df.columns = df.columns.str.strip().str.strip('"')
    df["alpha"]       = df["alpha"].apply(parse_mathematica_number)
    df["CF_Gamma"]    = df["CF_Gamma"].apply(parse_mathematica_number)
    df["Hydro_Gamma"] = df["Hydro_Gamma"].apply(parse_mathematica_number)
    df = df.dropna(subset=["alpha"])
    return df


# ── Plot setup ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}"
})

fig, ax = plt.subplots(figsize=(7, 5.5))

for filepath, colour in zip(FILES, COLOURS):
    label = parse_quantum_numbers(filepath)
    df = load_file(filepath)

    # CF data: positive values only
    df_cf = df[df["CF_Gamma"] > 0].dropna(subset=["CF_Gamma"])
    if not df_cf.empty:
        ax.plot(df_cf["alpha"], df_cf["CF_Gamma"],
                color=colour, linestyle="solid", linewidth=1.5)

    # Hydro data: positive values only - negative silently dropped
    df_hydro = df[df["Hydro_Gamma"] > 0].dropna(subset=["Hydro_Gamma"])
    if not df_hydro.empty:
        ax.plot(df_hydro["alpha"], df_hydro["Hydro_Gamma"],
                color=colour, linestyle="dashed", linewidth=1.5)

    # Label at peak of CF curve
    if not df_cf.empty:
        peak_idx = df_cf["CF_Gamma"].idxmax()
        peak_alpha = df_cf.loc[peak_idx, "alpha"]
        peak_gamma = df_cf.loc[peak_idx, "CF_Gamma"]
        ax.text(peak_alpha, peak_gamma * 1.5, label,
                color=colour, fontsize=11, ha="center", va="bottom")

# ── Legend: black lines only ──────────────────────────────────────────────────
legend_handles = [
    mlines.Line2D([], [], color="black", linestyle="solid",
                  linewidth=1.5, label="Continued Fractions"),
    mlines.Line2D([], [], color="black", linestyle="dashed",
                  linewidth=1.5, label="Non-Relativistic"),
]
ax.legend(handles=legend_handles, fontsize=10, frameon=False)

ax.set_yscale("log")
ax.set_xlabel(r"$\alpha$", fontsize=13)
ax.set_ylabel(r"$\Gamma^{\mathrm{sr}} r_g$", fontsize=13)
ax.grid(True, which="both", linestyle="--", alpha=0.4)

plt.tight_layout()

output_path = Path("Plots/superradiance_plot.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150)
plt.show()