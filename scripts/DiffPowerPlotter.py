import numpy as np
import matplotlib.pyplot as plt


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
# ANGULAR GRID
# ============================================================

theta = np.linspace(0, np.pi, 500)
phi = np.linspace(0, 2*np.pi, 1000)
theta_grid, phi_grid = np.meshgrid(theta, phi)

lon = phi_grid - np.pi
lat = np.pi/2 - theta_grid


# ============================================================
# ---- PLOT 1: 3d LEVEL ----
# ============================================================

power_3d = dP_3d(theta_grid)

fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.add_subplot(111, projection='mollweide')


im1 = ax1.pcolormesh(lon, lat, power_3d, shading='auto', cmap='inferno')
ax1.grid(True)
ax1.set_xticklabels(
    ['150°W','120°W','90°W','60°W','30°W','0°',
     '30°E','60°E','90°E','120°E','150°E']
)



cbar1 = fig1.colorbar(im1, ax=ax1, orientation='vertical', pad=0.05)
cbar1.set_label('dP/dΩ (3d level, natural units)')

fig1.savefig("3d.png", dpi=300, bbox_inches="tight")

plt.show()


# ============================================================
# ---- PLOT 2: 4f LEVEL ----
# ============================================================

power_4f = dP_4f(theta_grid)

fig2 = plt.figure(figsize=(10, 5))
ax2 = fig2.add_subplot(111, projection='mollweide')

im2 = ax2.pcolormesh(lon, lat, power_4f, shading='auto', cmap='inferno')
ax2.grid(True)
ax2.set_xticklabels(
    ['150°W','120°W','90°W','60°W','30°W','0°',
     '30°E','60°E','90°E','120°E','150°E']
)



cbar2 = fig2.colorbar(im2, ax=ax2, orientation='vertical', pad=0.05)
cbar2.set_label('dP/dΩ (4f level, natural units)')

fig2.savefig("4f.png", dpi=300, bbox_inches="tight")

plt.show()
