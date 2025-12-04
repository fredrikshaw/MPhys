import numpy as np
import matplotlib.pyplot as plt

# Define the full differential power function with parameters
def dP_dOmega(theta, rg=1.0, alpha=1.0, N=1.0):
    """
    Differential power dP/dΩ as a function of theta,
    with optional parameters rg, alpha, and N.
    Uses natural units with G_N = 1.
    """
    GN = 1.0  # natural units
    factor = 1 / (121 * np.pi * rg**4 * (324 + alpha**4)**12)
    amplitude = 5184 * GN * N**2 * alpha**20 * ( -18 + alpha**2 )**6
    bracket = (1154736 - 326592*alpha**2 + 24408*alpha**4 - 1008*alpha**6 + 11*alpha**8)**2
    angular = (35 + 28 * np.cos(2*theta) + np.cos(4*theta)) * np.sin(theta)**4
    return factor * amplitude * bracket * angular

# Create a grid of theta and phi
theta = np.linspace(0, np.pi, 500)       # polar angle
phi = np.linspace(0, 2*np.pi, 1000)      # azimuthal angle
theta_grid, phi_grid = np.meshgrid(theta, phi)

# Evaluate the power on the grid with default parameters (rg=1, alpha=1)
power = dP_dOmega(theta_grid, rg=1.0, alpha=1.0, N=1.0)

# Mollweide projection coordinates
lon = phi_grid - np.pi
lat = np.pi/2 - theta_grid

# Plotting
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection='mollweide')
im = ax.pcolormesh(lon, lat, power, shading='auto', cmap='inferno')
ax.grid(True)
ax.set_xticklabels(['150°W','120°W','90°W','60°W','30°W','0°','30°E','60°E','90°E','120°E','150°E'], fontsize=10)

# Colorbar on the right
cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.05)
cbar.set_label(r'$\frac{dP}{d\Omega} N^{-2}$')

plt.show()
