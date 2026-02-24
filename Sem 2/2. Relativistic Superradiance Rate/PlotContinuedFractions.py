from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re

def parse_mathematica_number(s):
    """Convert Mathematica number strings like 4.781`16.*^-10 to Python floats."""
    if isinstance(s, float):
        return s
    s = str(s).strip().strip('"')
    # Remove backtick precision marker and everything between backtick and *
    # e.g. 4.7811863`16.*^-10 -> 4.7811863e-10
    s = re.sub(r'`[\d.]*\.\*\^', 'e', s)
    # Handle plain *^-10 notation without backtick
    s = re.sub(r'\*\^', 'e', s)
    try:
        return float(s)
    except ValueError:
        return None

# Load file
data_path = "Mathematica/SR_n2l1m1_at0.99_aMin0.10_aMax0.50.dat"
df = pd.read_csv(data_path, sep="\t")

# Strip whitespace and quotes from column names
df.columns = df.columns.str.strip().str.strip('"')

print("Columns:", df.columns.tolist())
print(df.head())

# Parse all columns
df["alpha"]     = df["alpha"].apply(parse_mathematica_number)
df["CF_Gamma"]  = df["CF_Gamma"].apply(parse_mathematica_number)
df["Hydro_Gamma"] = df["Hydro_Gamma"].apply(parse_mathematica_number)

# Drop any rows that failed to parse
df = df.dropna()

print("\nParsed data:")
print(df.head())

# Plot
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}"
})
fig, ax = plt.subplots(figsize=(5.5, 5))

ax.plot(df["alpha"], df["CF_Gamma"],    color="#7842f5", linestyle="solid", linewidth=1.5, label="CF $\\Gamma$")
ax.plot(df["alpha"], df["Hydro_Gamma"], color="#7842f5", linestyle="dashed",       linewidth=1.5, label="Hydrogen-like $\\Gamma$")

ax.set_yscale("log")
ax.set_xlabel(r"$\alpha$", fontsize=13)
ax.set_ylabel(r"$\Gamma^\textrm{sr} r_g$", fontsize=13)
#ax.set_title(r"$|211\rangle$   $\tilde{a} = 0.99$", fontsize=13)
ax.legend(fontsize=11, frameon=False)
ax.grid(True, which="both", linestyle="--", alpha=0.5)

plt.text(0.4, 2e-8, r"$|211\rangle$", color='#7842f5')

plt.tight_layout()
plt.savefig("Plots/superradiance_plot.png", dpi=150)
plt.show()