"""
Quick Start Guide - Detector Distance Reach Plotter
====================================================

This guide shows you the most common use cases.
"""

# =============================================================================
# BASIC USAGE
# =============================================================================

from DetectorDistanceReach import plot_distance_reach

# Most basic usage - single annihilation process
plot_distance_reach(
    h_det=1e-24,        # Detection threshold
    alpha=0.1,          # Fine structure constant
    process='2p'        # Process (annihilation or transition)
)


# =============================================================================
# CUSTOMIZING PLOTS
# =============================================================================

# With custom parameters
plot_distance_reach(
    h_det=1e-24,
    alpha=0.1,
    process='2p',
    bh_mass_range=(0.1, 50),     # BH mass range in solar masses
    num_points=100,               # Number of points in plot
    delta_a_star=0.01,            # For annihilation processes
    save_path='my_plot.png',      # Save the plot
    show_plot=True                # Display the plot
)

# For transition processes
plot_distance_reach(
    h_det=1e-24,
    alpha=0.1,
    process='3p 2p',              # Note the space for transitions
    bh_mass_range=(0.1, 50),
    a_star=0.9,                   # Spin parameter for transitions
    save_path='transition_plot.png'
)


# =============================================================================
# COMPARING MULTIPLE PROCESSES
# =============================================================================

from DetectorDistanceReach import plot_multiple_processes

# Compare several processes at once
plot_multiple_processes(
    h_det=1e-24,
    alpha=0.1,
    processes=['2p', '3p', '3d', '3p 2p'],  # Mix of ann. and trans.
    bh_mass_range=(0.1, 50),
    save_path='comparison.png'
)


# =============================================================================
# CALCULATING SPECIFIC VALUES
# =============================================================================

from DetectorDistanceReach import calc_distance_reach_ann, calc_distance_reach_trans

# Calculate distance for a specific black hole mass
distance_ann = calc_distance_reach_ann(
    h_det=1e-24,
    alpha=0.1,
    process='2p',
    bh_mass_solar=1.0      # 1 solar mass
)
print(f"Distance reach for 1 M_☉: {distance_ann:.4e} kpc")

# For transition
distance_trans = calc_distance_reach_trans(
    h_det=1e-24,
    alpha=0.1,
    process='3p 2p',
    bh_mass_solar=1.0,
    a_star=0.9
)
print(f"Distance reach for transition: {distance_trans:.4e} kpc")


# =============================================================================
# PARAMETER STUDIES
# =============================================================================

import numpy as np

# Study effect of alpha
alphas = [0.05, 0.1, 0.15, 0.2]
for alpha in alphas:
    d = calc_distance_reach_ann(1e-24, alpha, '2p', 1.0)
    print(f"α={alpha:.2f}: {d:.4e} kpc")

# Study effect of detector sensitivity
h_dets = [1e-23, 1e-24, 1e-25]
for h_det in h_dets:
    d = calc_distance_reach_ann(h_det, 0.1, '2p', 1.0)
    print(f"h_det={h_det:.1e}: {d:.4e} kpc")

# Study effect of BH mass
masses = np.logspace(-1, 2, 10)  # 0.1 to 100 M_☉
for mass in masses:
    d = calc_distance_reach_ann(1e-24, 0.1, '2p', mass)
    print(f"M_BH={mass:.2f} M_☉: {d:.4e} kpc")


# =============================================================================
# AVAILABLE PROCESSES
# =============================================================================

# Annihilation processes (no space in name):
annihilation_processes = [
    '2p', '3p', '3d', '4p', '4d', '4f',
    '5p', '5d', '5f', '5g',
    '6p', '6d', '6f', '6g', '6h'
]

# Transition processes (space between levels):
transition_processes = [
    '3p 2p', '4p 2p', '4p 3p', '4d 3d',
    '5p 4p', '5d 4d', '5f 4f',
    '6g 5g', '7h 6h', '8i 7i', '9k 8k'
    # ... and many more
]

# Test any process
for proc in ['2p', '3d', '3p 2p']:
    try:
        plot_distance_reach(1e-24, 0.1, proc, show_plot=False,
                          save_path=f'test_{proc.replace(" ", "_")}.png')
        print(f"✓ Process '{proc}' works!")
    except Exception as e:
        print(f"✗ Process '{proc}' error: {e}")


# =============================================================================
# TYPICAL RESEARCH WORKFLOW
# =============================================================================

# Step 1: Set your parameters
h_det = 1e-24        # Your detector sensitivity
alpha = 0.1          # Your coupling constant
process = '2p'       # Your process of interest

# Step 2: Create overview plot
bh_masses, distances = plot_distance_reach(
    h_det=h_det,
    alpha=alpha,
    process=process,
    bh_mass_range=(0.1, 100),
    num_points=200,
    save_path=f'{process}_overview.png',
    show_plot=False
)

# Step 3: Find interesting mass range
max_distance_idx = np.argmax(distances)
optimal_mass = bh_masses[max_distance_idx]
print(f"Optimal BH mass: {optimal_mass:.2f} M_☉")

# Step 4: Calculate precise value
distance = calc_distance_reach_ann(h_det, alpha, process, optimal_mass)
print(f"Maximum distance reach: {distance:.4e} kpc")

# Step 5: Compare with other processes
comparison_processes = ['2p', '3p', '3d']
plot_multiple_processes(
    h_det=h_det,
    alpha=alpha,
    processes=comparison_processes,
    bh_mass_range=(0.1, 100),
    save_path='final_comparison.png'
)

print("Analysis complete!")


# =============================================================================
# ADVANCED USAGE
# =============================================================================

# Custom parameters for annihilation
from DetectorDistanceReach import calc_distance_reach_ann

distance = calc_distance_reach_ann(
    h_det=1e-24,
    alpha=0.1,
    process='2p',
    bh_mass_solar=1.0,
    delta_a_star=0.05,      # Larger spin difference
    m_quantum=1             # Azimuthal quantum number
)

# Custom parameters for transition
from DetectorDistanceReach import calc_distance_reach_trans

distance = calc_distance_reach_trans(
    h_det=1e-24,
    alpha=0.1,
    process='3p 2p',
    bh_mass_solar=1.0,
    a_star=0.95,           # Higher spin
    m_quantum=1            # Azimuthal quantum number
)


# =============================================================================
# ERROR HANDLING
# =============================================================================

try:
    # Invalid process name
    plot_distance_reach(1e-24, 0.1, 'invalid_process')
except ValueError as e:
    print(f"Caught error: {e}")

try:
    # Process exists but wrong type
    calc_distance_reach_ann(1e-24, 0.1, '3p 2p', 1.0)  # This is transition, not ann
except ValueError as e:
    print(f"Caught error: {e}")


# =============================================================================
# TIPS AND TRICKS
# =============================================================================

# Tip 1: Use log-spaced mass ranges for better coverage
bh_mass_range = (10**(-1), 10**2)  # 0.1 to 100 M_☉

# Tip 2: Higher num_points for smoother plots
plot_distance_reach(1e-24, 0.1, '2p', num_points=500)

# Tip 3: Return data for further analysis
masses, distances = plot_distance_reach(
    1e-24, 0.1, '2p', 
    show_plot=False,
    save_path='data_plot.png'
)
# Now you can analyze masses and distances arrays

# Tip 4: Batch process multiple alphas
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 7))
for alpha in [0.05, 0.1, 0.2]:
    masses, distances = plot_distance_reach(
        1e-24, alpha, '2p',
        show_plot=False,
        save_path=None
    )
    plt.loglog(masses, distances, label=f'α={alpha}')

plt.xlabel('Black Hole Mass [M_☉]')
plt.ylabel('Distance Reach [kpc]')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.savefig('alpha_comparison.png')
plt.close()

print("Custom comparison plot created!")


# =============================================================================
# REMEMBER
# =============================================================================

"""
Key points to remember:

1. Process names:
   - Annihilation: '2p', '3d', etc. (no space)
   - Transition: '3p 2p', '5f 4f', etc. (with space)

2. Units:
   - h_det: dimensionless (strain)
   - alpha: dimensionless (fine structure constant)
   - bh_mass_solar: solar masses
   - Output distance: kiloparsecs (kpc)

3. Default parameters:
   - delta_a_star = 0.01 (for annihilation)
   - a_star = 0.9 (for transitions)
   - m_quantum = 1
   - num_points = 100

4. Common ranges:
   - h_det: 1e-26 to 1e-23
   - alpha: 0.01 to 0.5
   - BH mass: 0.001 to 1000 M_☉

5. For help:
   - See README.md for full documentation
   - Run DetectorDistanceReach.py for built-in tests
   - Run extended_tests.py for more examples
   - Run interactive_examples.py for guided tour
"""
