"""
Additional tests and examples for DetectorDistanceReach.py

This script demonstrates various use cases and parameter configurations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '0. Scripts from Sem 1'))

from DetectorDistanceReach import (
    plot_distance_reach,
    plot_multiple_processes,
    calc_distance_reach_ann,
    calc_distance_reach_trans
)
import numpy as np

print("=" * 80)
print("Extended Test Suite for Detector Distance Reach")
print("=" * 80)
print()

# Test 5: Different alpha values for same process
print("-" * 80)
print("Test 5: Effect of Different Alpha Values (2p annihilation)")
print("-" * 80)

h_det = 1e-24
alphas = [0.05, 0.1, 0.2]
bh_mass = 1.0  # Solar mass

for alpha in alphas:
    try:
        distance = calc_distance_reach_ann(h_det, alpha, '2p', bh_mass)
        print(f"  α = {alpha:.2f}: Distance reach = {distance:.4e} kpc")
    except Exception as e:
        print(f"  α = {alpha:.2f}: Error - {e}")

print()

# Test 6: Different detection thresholds
print("-" * 80)
print("Test 6: Effect of Detection Threshold (3d annihilation, α=0.1)")
print("-" * 80)

alpha = 0.1
h_dets = [1e-23, 1e-24, 1e-25]
bh_mass = 1.0e-6

for h_det in h_dets:
    try:
        distance = calc_distance_reach_ann(h_det, alpha, '3d', bh_mass)
        print(f"  h_det = {h_det:.1e}: Distance reach = {distance:.4e} kpc")
    except Exception as e:
        print(f"  h_det = {h_det:.1e}: Error - {e}")

print()

# Test 7: Transition processes with different parameters
print("-" * 80)
print("Test 7: Transition Processes (α=0.1, h_det=1e-24)")
print("-" * 80)

h_det = 1e-24
alpha = 0.1
transitions = ['3p 2p', '4p 3p', '5f 4f']
bh_mass = 1.0

for trans in transitions:
    try:
        distance = calc_distance_reach_trans(h_det, alpha, trans, bh_mass)
        print(f"  {trans}: Distance reach = {distance:.4e} kpc")
    except Exception as e:
        print(f"  {trans}: Error - {e}")

print()

# Test 8: High-level transitions
print("-" * 80)
print("Test 8: High-Level Transitions (α=0.1, h_det=1e-24)")
print("-" * 80)

h_det = 1e-24
alpha = 0.1
high_transitions = ['6g 5g', '7h 6h']
bh_mass = 1.0

for trans in high_transitions:
    try:
        distance = calc_distance_reach_trans(h_det, alpha, trans, bh_mass, a_star=0.9)
        print(f"  {trans}: Distance reach = {distance:.4e} kpc")
    except Exception as e:
        print(f"  {trans}: Error - {e}")

print()

# Test 9: Higher-order annihilation processes
print("-" * 80)
print("Test 9: Higher-Order Annihilation Processes (α=0.1, h_det=1e-24)")
print("-" * 80)

h_det = 1e-24
alpha = 0.1
ann_processes = ['4f', '5g', '6h']
bh_mass = 1.0

for proc in ann_processes:
    try:
        distance = calc_distance_reach_ann(h_det, alpha, proc, bh_mass)
        print(f"  {proc}: Distance reach = {distance:.4e} kpc")
    except Exception as e:
        print(f"  {proc}: Error - {e}")

print()

# Test 10: Varying black hole mass
print("-" * 80)
print("Test 10: Effect of Black Hole Mass (2p, α=0.1, h_det=1e-24)")
print("-" * 80)

h_det = 1e-24
alpha = 0.1
process = '2p'
bh_masses_test = [0.1, 1.0, 10.0, 50.0]

for bh_mass in bh_masses_test:
    try:
        distance = calc_distance_reach_ann(h_det, alpha, process, bh_mass)
        print(f"  M_BH = {bh_mass:5.1f} M_☉: Distance reach = {distance:.4e} kpc")
    except Exception as e:
        print(f"  M_BH = {bh_mass:5.1f} M_☉: Error - {e}")

print()

# Test 11: Create comprehensive comparison plot
print("-" * 80)
print("Test 11: Creating Comprehensive Comparison Plot")
print("-" * 80)

try:
    # Compare multiple annihilation and transition processes
    mixed_processes = ['2p', '3p', '3d', '3p 2p']
    h_det = 1e-24
    alpha = 0.1
    
    results = plot_multiple_processes(
        h_det=h_det,
        alpha=alpha,
        processes=mixed_processes,
        bh_mass_range=(0.1, 50),
        num_points=100,
        save_path='comprehensive_distance_comparison.png',
        show_plot=False
    )
    
    print("✓ Successfully created comprehensive comparison plot")
    print(f"  Processes compared: {', '.join(mixed_processes)}")
    print(f"  Plot saved as: comprehensive_distance_comparison.png")
    
except Exception as e:
    print(f"✗ Error: {e}")

print()

# Test 12: Different spin parameters for transitions
print("-" * 80)
print("Test 12: Effect of Spin Parameter on Transitions (3p 2p, α=0.1)")
print("-" * 80)

h_det = 1e-24
alpha = 0.1
process = '3p 2p'
bh_mass = 1.0
a_stars = [0.5, 0.7, 0.9, 0.99]

for a_star in a_stars:
    try:
        distance = calc_distance_reach_trans(h_det, alpha, process, bh_mass, a_star=a_star)
        print(f"  a* = {a_star:.2f}: Distance reach = {distance:.4e} kpc")
    except Exception as e:
        print(f"  a* = {a_star:.2f}: Error - {e}")

print()

# Test 13: Different delta_a_star for annihilation
print("-" * 80)
print("Test 13: Effect of Δa* on Annihilation (2p, α=0.1)")
print("-" * 80)

h_det = 1e-24
alpha = 0.1
process = '2p'
bh_mass = 1.0
delta_a_stars = [0.001, 0.01, 0.1]

for delta_a_star in delta_a_stars:
    try:
        distance = calc_distance_reach_ann(h_det, alpha, process, bh_mass, delta_a_star=delta_a_star)
        print(f"  Δa* = {delta_a_star:.3f}: Distance reach = {distance:.4e} kpc")
    except Exception as e:
        print(f"  Δa* = {delta_a_star:.3f}: Error - {e}")

print()

print("=" * 80)
print("Extended test suite completed!")
print("=" * 80)
print()
print("Summary of outputs:")
print("  - All test results displayed above")
print("  - Additional plot: comprehensive_distance_comparison.png")
print()
print("The script successfully demonstrates:")
print("  ✓ Distance calculations for annihilation processes")
print("  ✓ Distance calculations for transition processes")
print("  ✓ Effects of varying α, h_det, M_BH, a*, and Δa*")
print("  ✓ Comparison of multiple processes")
