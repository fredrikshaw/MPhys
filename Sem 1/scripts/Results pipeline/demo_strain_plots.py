"""
Demonstration of StrainVsFrequencyPlot for both annihilations and transitions.

This script shows how to use the updated StrainVsFrequencyPlot module to generate
plots for both annihilation and transition processes.
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch processing

from StrainVsFrequencyPlot import plot_strain_vs_frequency

# Common parameters
bh_mass_solar = 1e-11  # Black hole mass in solar masses
alpha = 0.1           # Fine structure constant
astar_init = 0.687    # Initial spin parameter
distance_kpc = 10     # Distance in kpc
excluded_processes_strain = ["8p 3p", "7d 3d", "7p 2p", "5p 2p", "6p 2p", "6p 3p", "6d 3d", "7p 5p", "7d 5d", "7f 5f", "6f 4f", "6d 4d", "6p 4p", "7p 4p", "7d 4d", "4p 2p", "4p 3p", "7p 3p", "5d 3d"]  # Excluded transitions
excluded_processes_rate = ["7g 6g", "8d 3d", "5f 4f", "6d 5d", "6f 5f", "5p 2p", "6p 2p", "6p 3p", "6d 3d", "7p 5p", "7d 5d", "7f 5f", "6f 4f", "6d 4d", "6p 4p", "7p 4p", "7d 4d", "4p 2p", "4p 3p", "7p 3p", "5d 3d"]  # Excluded transitions

print("=" * 80)
print("STRAIN VS FREQUENCY PLOT DEMONSTRATION")
print("=" * 80)
print(f"\nParameters:")
print(f"  Black hole mass: {bh_mass_solar} M☉")
print(f"  Alpha: {alpha}")
print(f"  Initial spin: {astar_init}")
print(f"  Distance: {distance_kpc} kpc\n")

# 1. Annihilation - Peak Strain
print("-" * 80)
print("1. ANNIHILATION - PEAK STRAIN")
print("-" * 80)
fig1, freqs1, strains1, rates1, procs1 = plot_strain_vs_frequency(
    bh_mass_solar, alpha, plot_type='strain', process='annihilation', 
    astar_init=astar_init, distance_kpc=distance_kpc
)
print(f"Found {len(procs1)} valid annihilation processes\n")

# 2. Annihilation - Annihilation Rate
print("-" * 80)
print("2. ANNIHILATION - ANNIHILATION RATE")
print("-" * 80)
fig2, freqs2, strains2, rates2, procs2 = plot_strain_vs_frequency(
    bh_mass_solar, alpha, plot_type='rate', process='annihilation',
    astar_init=astar_init, distance_kpc=distance_kpc
)
print(f"Found {len(procs2)} valid annihilation processes\n")

# 3. Transition - Peak Strain
print("-" * 80)
print("3. TRANSITION - PEAK STRAIN")
print("-" * 80)
fig3, freqs3, strains3, rates3, procs3 = plot_strain_vs_frequency(
    bh_mass_solar, alpha, plot_type='strain', process='transition',
    astar_init=astar_init, distance_kpc=distance_kpc, 
    exclude_processes=excluded_processes_strain
)
print(f"Found {len(procs3)} valid transition processes\n")

# 4. Transition - Transition Rate
print("-" * 80)
print("4. TRANSITION - TRANSITION RATE")
print("-" * 80)
fig4, freqs4, strains4, rates4, procs4 = plot_strain_vs_frequency(
    bh_mass_solar, alpha, plot_type='rate', process='transition',
    astar_init=astar_init, distance_kpc=distance_kpc, 
    exclude_processes=excluded_processes_rate
)
print(f"Found {len(procs4)} valid transition processes\n")

print("=" * 80)
print("ALL PLOTS GENERATED SUCCESSFULLY")
print("=" * 80)
print("\nPlots saved in: Final plots/")
print("  - AnnihilationStrainVsFrequency_*.pdf")
print("  - AnnihilationRateVsFrequency_*.pdf")
print("  - TransitionStrainVsFrequency_*.pdf")
print("  - TransitionRateVsFrequency_*.pdf")
