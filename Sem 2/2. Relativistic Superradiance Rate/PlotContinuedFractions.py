import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import re


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
    "Sem 2/2. Relativistic Superradiance Rate/Mathematica/Data/SR_n5l4m4_at0.990_aMin0.010_aMax1.909_20260428.dat",
    "Sem 2/2. Relativistic Superradiance Rate/Mathematica/Data/SR_n2l1m1_at0.990_aMin0.010_aMax0.564_20260428.dat",
    "Sem 2/2. Relativistic Superradiance Rate/Mathematica/Data/SR_n5l4m4_at0.900_aMin0.600_aMax1.379_20260428.dat",
    #"Sem 2/2. Relativistic Superradiance Rate/Mathematica/SR_n5l4m4_at0.990_aMin0.010_aMax2.100_20260312.dat",
]


COLOURS = ["#68308d", "#a31818","#356E44", "#43d5fa", "#284245", "green"]
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


# ── Plot setup ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}"
})

fig, ax = plt.subplots(figsize=(4, 3.5))

for filepath, colour in zip(FILES, COLOURS):
    label, quantum_numbers = parse_quantum_numbers(filepath)
    df = load_file(filepath)

    if quantum_numbers: 
        n, m, l = quantum_numbers
        a_star = 0.99 # replace later

        # CF data: positive values only
        df_cf = df[df["CF_Gamma"] > 0].dropna(subset=["CF_Gamma"])
        if not df_cf.empty:
            ax.plot(df_cf["alpha"], df_cf["CF_Gamma"],
                    color=colour, linestyle="solid", linewidth=1.5)

        # Hydro data: replace with calculated values
        USE_CALCULATED_HYDRO = False

        df_hydro = df[df["Hydro_Gamma"] > 0].dropna(subset=["Hydro_Gamma"])
        if not df_hydro.empty:

            if USE_CALCULATED_HYDRO:
                df_hydro["Hydro_Gamma_Used"] = df_hydro["alpha"].apply(
                    lambda alpha: calc_superradiance_rate(l, m, n, a_star, 1, alpha)
                )
            else:
                df_hydro["Hydro_Gamma_Used"] = df_hydro["Hydro_Gamma"]

            ax.plot(df_hydro["alpha"], df_hydro["Hydro_Gamma_Used"],
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

#ax.legend(handles=legend_handles, fontsize=10, frameon=False)

ax.set_yscale("log")
ax.set_xlabel(r"$\alpha$", fontsize=13)
ax.set_ylabel(r"$\Gamma^{\mathrm{sr}} r_g$", fontsize=13)
ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0,), numticks=100))
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.set_ylim(1e-16, 1e-6)
ax.set_xlim(0, 2.1)

plt.tight_layout()

output_path = Path("Sem 2/2. Relativistic Superradiance Rate/Plots/superradiance_plot.pdf")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150)
plt.show()