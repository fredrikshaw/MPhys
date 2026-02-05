import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import constants

SOLAR_MASS = 1.988e30  # [kg]

blackholemass = 1e-11  # [solar masses]

m_bh_J = blackholemass * SOLAR_MASS * constants.c ** 2       # [J]
m_bh_ev = m_bh_J / constants.e                               # [eV]
G_N = 6.708e-57                                              # [eV^-2]
r_g = G_N * m_bh_ev                                          # [eV^-1]
r_g_SI = constants.G * blackholemass * SOLAR_MASS / (constants.c)**2     # [m]

N_max = 1e63


# ============================================================
# SETUP LaTeX FORMATTING
# ============================================================

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}"
})


# ============================================================
# 2p LEVEL DIFFERENTIAL POWER dP/dΩ
# ============================================================

def dP_2p(theta, rg=1.0, alpha=1.0, N=1.0):
    GN = 1.0  # natural units

    factor = 1 / (1225 * np.pi * rg**4 * (64 + alpha**4)**8)
    amplitude = 1024 * GN * N**2 * alpha**20 * (-8 + alpha**2)**2
    bracket = (832 - 96*alpha**2 + 13*alpha**4)**2
    angular = (35 + 28*np.cos(2*theta) + np.cos(4*theta))

    return factor * amplitude * bracket * angular


# ============================================================
# 3d LEVEL DIFFERENTIAL POWER dP/dΩ
# ============================================================

def dP_3d(theta, rg=1.0, alpha=1.0, N=1.0):
    GN = 1.0  # natural units

    factor = 1 / (121 * np.pi * rg**4 * (324 + alpha**4)**12)
    amplitude = 5184 * GN * N**2 * alpha**20 * (-18 + alpha**2)**6
    bracket = (1154736 - 326592*alpha**2 + 24408*alpha**4
               - 1008*alpha**6 + 11*alpha**8)**2
    angular = (35 + 28*np.cos(2*theta) + np.cos(4*theta)) * np.sin(theta)**4

    return factor * amplitude * bracket * angular


# ============================================================
# 4f LEVEL DIFFERENTIAL POWER dP/dΩ
# ============================================================

def dP_4f(theta, rg=1.0, alpha=1.0, N=1.0):
    GN = 1.0  # natural units

    factor = 1 / (169 * np.pi * rg**4 * (1024 + alpha**4)**16)
    amplitude = 1677721600 * GN * N**2 * alpha**24 * (-32 + alpha**2)**10
    bracket = (13631488 - 1900544*alpha**2 + 83968*alpha**4
               - 1856*alpha**6 + 13*alpha**8)**2
    angular = (35 + 28*np.cos(2*theta) + np.cos(4*theta)) * np.sin(theta)**8

    return factor * amplitude * bracket * angular


# ============================================================
# 6g→5g LEVEL DIFFERENTIAL POWER dP/dΩ
# ============================================================

def dP_6gto5g(theta, rg=1.0, alpha=1.0, N=1.0):
    GN = 1.0  # natural units
    
    # Precompute some terms to make the expression more readable
    alpha2 = alpha**2
    alpha4 = alpha**4
    alpha6 = alpha**6
    alpha8 = alpha**8
    alpha10 = alpha**10
    alpha12 = alpha**12
    
    # Main coefficient
    coeff = 3411634563591564165120000000000000000000000000 * GN * N**2 * alpha**12
    
    # Large denominator
    denominator = 173666556496502151585132390487441 * np.pi * rg**4 * (3600 + alpha2)**22
    
    # Complex bracket term
    # Constant term
    term0 = -8579703408427008000000000000 \
            - 39048650128097280000000000 * alpha2 \
            + 69358713451315200000000 * alpha4 \
            + 42885582917760000000 * alpha6 \
            + 6756330113280000 * alpha8 \
            + 601035033600 * alpha10 \
            + 190021546 * alpha12
    
    # Cos[2θ] term
    cos2theta = 11 * alpha2 * (-1416617695641600000000000 \
                               - 4892770676736000000000 * alpha2 \
                               + 1266509732544000000 * alpha4 \
                               + 1810374446880000 * alpha6 \
                               + 324444454800 * alpha8 \
                               + 20831239 * alpha10)
    
    # Cos[4θ] term
    cos4theta = 286 * alpha4 * (-32695629004800000000 \
                                - 30555527616000000 * alpha2 \
                                - 3069925920000 * alpha4 \
                                + 2346774000 * alpha6 \
                                + 898909 * alpha8)
    
    # Cos[6θ] term (first part)
    cos6theta_part1 = -1748967764544000000 * alpha6 \
                      - 671259869280000 * alpha8 \
                      - 152832279600 * alpha10 \
                      + 39122083 * alpha12
    
    # Cos[8θ] term
    cos8theta = -92185853760000 * alpha8 \
                - 25607181600 * alpha10
    
    # Put it all together
    bracket = (term0 \
               + cos2theta * np.cos(2*theta) \
               + cos4theta * np.cos(4*theta) \
               + cos6theta_part1 * np.cos(6*theta) \
               + cos8theta * np.cos(8*theta))**2
    
    # Angular part
    angular = np.sin(theta)**4
    
    return (coeff / denominator) * bracket * angular


# ============================================================
# ANGULAR GRID
# ============================================================

theta = np.linspace(0, np.pi, 250)
phi = np.linspace(0, 2*np.pi, 2)
theta_grid, phi_grid = np.meshgrid(theta, phi)

lon = phi_grid - np.pi
lat = np.pi/2 - theta_grid


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

output_dir = "scripts/DiffPowerPlots"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# ---- PLOT 1: 2p LEVEL (Normalised to 1) ----
# ============================================================

power_2p = dP_2p(theta_grid, rg=r_g, alpha=0.1, N=N_max)

# Normalize to maximum value = 1
power_2p_normalized = power_2p / np.max(power_2p)

fig1 = plt.figure(figsize=(5, 2.5))
ax1 = fig1.add_subplot(111, projection='mollweide')

im1 = ax1.pcolormesh(lon, lat, power_2p_normalized, shading='auto', cmap='inferno', vmin=0, vmax=1, rasterized=True)
ax1.grid(True)
ax1.set_xticklabels([])  # Remove x-tick labels

cbar1 = fig1.colorbar(im1, ax=ax1, orientation='vertical', pad=0.05)
cbar1.set_label(r'$\text{d}P/\text{d}\Omega$ (Normalised)')

fig1.savefig(os.path.join(output_dir, "2p_normalized.pdf"), bbox_inches="tight")
fig1.savefig(os.path.join(output_dir, "2p_normalized.png"), dpi=300, bbox_inches="tight")

plt.show()


# ============================================================
# ---- PLOT 2: 3d LEVEL (Normalised to 1) ----
# ============================================================

power_3d = dP_3d(theta_grid, rg=r_g, alpha=0.1, N=N_max)

# Normalize to maximum value = 1
power_3d_normalized = power_3d / np.max(power_3d)

fig2 = plt.figure(figsize=(5, 2.5))
ax2 = fig2.add_subplot(111, projection='mollweide')

im2 = ax2.pcolormesh(lon, lat, power_3d_normalized, shading='auto', cmap='inferno', vmin=0, vmax=1, rasterized=True)
ax2.grid(True)
ax2.set_xticklabels([])  # Remove x-tick labels

cbar2 = fig2.colorbar(im2, ax=ax2, orientation='vertical', pad=0.05)
cbar2.set_label(r'$\text{d}P/\text{d}\Omega$ (Normalised)')

fig2.savefig(os.path.join(output_dir, "3d_normalized.pdf"), bbox_inches="tight")
fig2.savefig(os.path.join(output_dir, "3d_normalized.png"), dpi=300, bbox_inches="tight")

plt.show()


# ============================================================
# ---- PLOT 3: 4f LEVEL (Normalised to 1) ----
# ============================================================

power_4f = dP_4f(theta_grid, rg=r_g, alpha=0.1, N=N_max)

# Normalize to maximum value = 1
power_4f_normalized = power_4f / np.max(power_4f)

fig3 = plt.figure(figsize=(5, 2.5))
ax3 = fig3.add_subplot(111, projection='mollweide')

im3 = ax3.pcolormesh(lon, lat, power_4f_normalized, shading='auto', cmap='inferno', vmin=0, vmax=1, rasterized=True)
ax3.grid(True)
ax3.set_xticklabels([])  # Remove x-tick labels

cbar3 = fig3.colorbar(im3, ax=ax3, orientation='vertical', pad=0.05)
cbar3.set_label(r'$\text{d}P/\text{d}\Omega$ (Normalised)')

fig3.savefig(os.path.join(output_dir, "4f_normalized.pdf"), bbox_inches="tight")
fig3.savefig(os.path.join(output_dir, "4f_normalized.png"), dpi=300, bbox_inches="tight")

plt.show()


# ============================================================
# ---- PLOT 4: 6g→5g LEVEL (Normalised to 1) ----
# ============================================================

power_6gto5g = dP_6gto5g(theta_grid, rg=r_g, alpha=0.1, N=N_max)

# Normalize to maximum value = 1
power_6gto5g_normalized = power_6gto5g / np.max(power_6gto5g)

fig4 = plt.figure(figsize=(5, 2.5))
ax4 = fig4.add_subplot(111, projection='mollweide')

im4 = ax4.pcolormesh(lon, lat, power_6gto5g_normalized, shading='auto', cmap='inferno', vmin=0, vmax=1, rasterized=True)
ax4.grid(True)
ax4.set_xticklabels([])  # Remove x-tick labels

cbar4 = fig4.colorbar(im4, ax=ax4, orientation='vertical', pad=0.05)
cbar4.set_label(r'$\text{d}P/\text{d}\Omega$ (Normalised)')

fig4.savefig(os.path.join(output_dir, "6gto5g_normalized.pdf"), bbox_inches="tight")
fig4.savefig(os.path.join(output_dir, "6gto5g_normalized.png"), dpi=300, bbox_inches="tight")

plt.show()