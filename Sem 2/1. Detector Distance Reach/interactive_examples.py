"""
Interactive Examples - Detector Distance Reach

This script provides practical examples for using the DetectorDistanceReach module
in research scenarios.
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
import matplotlib.pyplot as plt

def example_1_ligo_sensitivity():
    """
    Example 1: Calculate reach for LIGO sensitivity
    """
    print("=" * 80)
    print("Example 1: LIGO-like Detector Sensitivity")
    print("=" * 80)
    print()
    print("Scenario: Advanced LIGO detector with h_det ~ 1e-24")
    print("Question: What is the distance reach for different BH masses?")
    print()
    
    h_det = 1e-24
    alpha = 0.1
    process = '2p'
    
    print(f"Parameters: h_det={h_det:.1e}, α={alpha}, process={process}")
    print()
    
    # Calculate for various masses
    masses = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    print("Black Hole Mass [M_☉]  |  Distance Reach [kpc]  |  Distance Reach [pc]")
    print("-" * 80)
    
    for mass in masses:
        dist_kpc = calc_distance_reach_ann(h_det, alpha, process, mass)
        dist_pc = dist_kpc * 1000
        print(f"{mass:20.1f}  |  {dist_kpc:20.4e}  |  {dist_pc:18.4e}")
    
    print()
    print("Conclusion: Distance reach scales linearly with BH mass for annihilation.")
    print()


def example_2_compare_detectors():
    """
    Example 2: Compare different detector sensitivities
    """
    print("=" * 80)
    print("Example 2: Comparing Different Detector Sensitivities")
    print("=" * 80)
    print()
    print("Scenario: How does detector sensitivity affect reach?")
    print()
    
    alpha = 0.1
    process = '2p'
    bh_mass = 1.0  # Solar mass
    
    detectors = {
        'LIGO (current)': 1e-23,
        'Advanced LIGO': 1e-24,
        'Einstein Telescope': 1e-25,
        'Future detector': 1e-26
    }
    
    print(f"Process: {process}, α={alpha}, M_BH={bh_mass} M_☉")
    print()
    print("Detector               |  h_det     |  Distance Reach [kpc]  |  Reach [pc]")
    print("-" * 80)
    
    for name, h_det in detectors.items():
        dist_kpc = calc_distance_reach_ann(h_det, alpha, process, bh_mass)
        dist_pc = dist_kpc * 1000
        print(f"{name:22} |  {h_det:.1e}  |  {dist_kpc:20.4e}  |  {dist_pc:10.4e}")
    
    print()
    print("Conclusion: 10x better sensitivity → 10x greater distance reach.")
    print()


def example_3_alpha_dependence():
    """
    Example 3: Study alpha dependence
    """
    print("=" * 80)
    print("Example 3: Effect of Fine Structure Constant (α)")
    print("=" * 80)
    print()
    print("Scenario: How does α affect the distance reach?")
    print()
    
    h_det = 1e-24
    process = '2p'
    bh_mass = 1.0
    
    alphas = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    
    print(f"Process: {process}, h_det={h_det:.1e}, M_BH={bh_mass} M_☉")
    print()
    print("α        |  Distance Reach [kpc]  |  Distance Reach [pc]")
    print("-" * 60)
    
    distances = []
    for alpha in alphas:
        dist_kpc = calc_distance_reach_ann(h_det, alpha, process, bh_mass)
        dist_pc = dist_kpc * 1000
        distances.append(dist_kpc)
        print(f"{alpha:8.2f} |  {dist_kpc:20.4e}  |  {dist_pc:18.4e}")
    
    print()
    print("Conclusion: Larger α → stronger coupling → larger distance reach.")
    print()


def example_4_annihilation_vs_transition():
    """
    Example 4: Compare annihilation vs transition
    """
    print("=" * 80)
    print("Example 4: Annihilation vs Transition Processes")
    print("=" * 80)
    print()
    print("Scenario: Which process gives better detectability?")
    print()
    
    h_det = 1e-24
    alpha = 0.1
    bh_mass = 1.0
    
    ann_process = '3p'
    trans_process = '3p 2p'
    
    dist_ann = calc_distance_reach_ann(h_det, alpha, ann_process, bh_mass)
    dist_trans = calc_distance_reach_trans(h_det, alpha, trans_process, bh_mass)
    
    print(f"Parameters: h_det={h_det:.1e}, α={alpha}, M_BH={bh_mass} M_☉")
    print()
    print(f"Annihilation ({ann_process}):")
    print(f"  Distance reach: {dist_ann:.4e} kpc = {dist_ann*1000:.4e} pc")
    print()
    print(f"Transition ({trans_process}):")
    print(f"  Distance reach: {dist_trans:.4e} kpc = {dist_trans*1000:.4e} pc")
    print()
    print(f"Ratio (Transition/Annihilation): {dist_trans/dist_ann:.4e}")
    print()
    print("Conclusion: Transitions can have much larger reach than annihilation!")
    print()


def example_5_create_publication_plot():
    """
    Example 5: Create publication-quality plot
    """
    print("=" * 80)
    print("Example 5: Creating Publication-Quality Comparison Plot")
    print("=" * 80)
    print()
    
    h_det = 1e-24
    alpha = 0.1
    
    # Select interesting processes
    processes = ['2p', '3p', '3d', '4f']
    
    print("Creating comparison plot for processes:", processes)
    print(f"Parameters: h_det={h_det:.1e}, α={alpha}")
    print()
    
    results = plot_multiple_processes(
        h_det=h_det,
        alpha=alpha,
        processes=processes,
        bh_mass_range=(0.1, 50),
        num_points=200,  # High resolution
        save_path='publication_quality_plot.png',
        show_plot=False
    )
    
    print("✓ Plot created: publication_quality_plot.png")
    print()
    
    # Print summary statistics
    print("Summary of Results:")
    print("-" * 80)
    print("Process  |  Min Distance [pc]  |  Max Distance [pc]  |  Range Factor")
    print("-" * 80)
    
    for proc, (masses, dists) in results.items():
        valid_dists = dists[~np.isnan(dists)] * 1000  # Convert to pc
        if len(valid_dists) > 0:
            min_dist = np.min(valid_dists)
            max_dist = np.max(valid_dists)
            range_factor = max_dist / min_dist
            print(f"{proc:8} |  {min_dist:18.4e}  |  {max_dist:18.4e}  |  {range_factor:12.2f}")
    
    print()


def example_6_orbital_level_comparison():
    """
    Example 6: Compare different orbital levels
    """
    print("=" * 80)
    print("Example 6: Effect of Orbital Angular Momentum")
    print("=" * 80)
    print()
    print("Scenario: How does l quantum number affect detectability?")
    print()
    
    h_det = 1e-24
    alpha = 0.1
    bh_mass = 1.0
    
    # Compare processes with same n but different l
    n3_processes = ['3p', '3d']  # n=3, l=1 and l=2
    n4_processes = ['4p', '4d', '4f']  # n=4, l=1,2,3
    n5_processes = ['5p', '5d', '5f', '5g']  # n=5, l=1,2,3,4
    
    print(f"Parameters: h_det={h_det:.1e}, α={alpha}, M_BH={bh_mass} M_☉")
    print()
    
    for n, processes in [(3, n3_processes), (4, n4_processes), (5, n5_processes)]:
        print(f"n={n} states:")
        print("  Process  |  l  |  Distance Reach [pc]")
        print("  " + "-" * 45)
        
        for proc in processes:
            l_map = {'p': 1, 'd': 2, 'f': 3, 'g': 4}
            l = l_map[proc[1]]
            dist_kpc = calc_distance_reach_ann(h_det, alpha, proc, bh_mass)
            dist_pc = dist_kpc * 1000
            print(f"  {proc:8} |  {l}  |  {dist_pc:25.4e}")
        print()
    
    print("Conclusion: Higher l (more angular momentum) → lower detectability.")
    print()


def example_7_practical_survey():
    """
    Example 7: Design a survey strategy
    """
    print("=" * 80)
    print("Example 7: Designing a Detection Survey")
    print("=" * 80)
    print()
    print("Scenario: You have a detector with h_det = 1e-24.")
    print("Question: What is the survey volume for different processes?")
    print()
    
    h_det = 1e-24
    alpha = 0.1
    bh_mass = 1.0
    
    processes_to_survey = ['2p', '3p', '3p 2p']
    
    print(f"Parameters: h_det={h_det:.1e}, α={alpha}, M_BH={bh_mass} M_☉")
    print()
    print("Process    |  Type        |  Reach [kpc]  |  Volume [kpc³]      ")
    print("-" * 75)
    
    for proc in processes_to_survey:
        proc_type = 'Annihilation' if ' ' not in proc else 'Transition'
        
        if proc_type == 'Annihilation':
            dist_kpc = calc_distance_reach_ann(h_det, alpha, proc, bh_mass)
        else:
            dist_kpc = calc_distance_reach_trans(h_det, alpha, proc, bh_mass)
        
        volume_kpc3 = (4/3) * np.pi * dist_kpc**3
        
        print(f"{proc:10} |  {proc_type:12} |  {dist_kpc:11.4e}  |  {volume_kpc3:18.4e}")
    
    print()
    print("Note: Larger survey volume → higher event rate (if merger rate is uniform).")
    print()


# Run all examples
if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  INTERACTIVE EXAMPLES - DETECTOR DISTANCE REACH  ".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Run each example
    example_1_ligo_sensitivity()
    input("Press Enter to continue to Example 2...")
    print("\n")
    
    example_2_compare_detectors()
    input("Press Enter to continue to Example 3...")
    print("\n")
    
    example_3_alpha_dependence()
    input("Press Enter to continue to Example 4...")
    print("\n")
    
    example_4_annihilation_vs_transition()
    input("Press Enter to continue to Example 5...")
    print("\n")
    
    example_5_create_publication_plot()
    input("Press Enter to continue to Example 6...")
    print("\n")
    
    example_6_orbital_level_comparison()
    input("Press Enter to continue to Example 7...")
    print("\n")
    
    example_7_practical_survey()
    
    print("\n")
    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  - publication_quality_plot.png")
    print()
    print("You can now use these functions in your own research!")
    print("See README.md for more detailed documentation.")
    print()
