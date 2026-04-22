# Detector Distance Reach Plotter

A Python module for calculating and plotting the distance reach of gravitational wave detectors for detecting signals from primordial black holes with gravitational atom processes.

## Overview

This module calculates the maximum distance at which a gravitational wave detector can detect signals from black holes undergoing either:
- **Annihilation processes** (e.g., 2p, 3d, 4f)
- **Transition processes** (e.g., 3p→2p, 5f→4f, 7h→6h)

The calculations use processes defined in `ConvertedFunctions.py` and utility functions from `ParamCalculator.py`.

## Features

- Calculate detector distance reach for annihilation processes
- Calculate detector distance reach for transition processes
- Plot distance reach vs black hole mass
- Compare multiple processes on the same plot
- Automatic unit conversions (eV⁻¹ to kpc)
- Process name validation
- Comprehensive error handling

## Installation

No special installation required. Just ensure you have the following dependencies:

```bash
pip install numpy matplotlib scipy
```

## Quick Start

```python
from DetectorDistanceReach import plot_distance_reach, calc_distance_reach_ann

# Plot distance reach for 2p annihilation
h_det = 1e-24  # Detection threshold
alpha = 0.1    # Fine structure constant

bh_masses, distances = plot_distance_reach(
    h_det=h_det,
    alpha=alpha,
    process='2p',
    bh_mass_range=(0.1, 50),  # Solar masses
    save_path='my_plot.png'
)

# Calculate distance for a specific black hole mass
distance_kpc = calc_distance_reach_ann(
    h_det=1e-24,
    alpha=0.1,
    process='2p',
    bh_mass_solar=1.0  # 1 solar mass
)
print(f"Distance reach: {distance_kpc:.4e} kpc")
```

## Usage Examples

### Example 1: Single Annihilation Process

```python
from DetectorDistanceReach import plot_distance_reach

# Plot 3d annihilation process
plot_distance_reach(
    h_det=1e-24,           # Detection threshold
    alpha=0.1,             # Fine structure constant
    process='3d',          # Process type
    bh_mass_range=(0.1, 100),  # BH mass range [M_☉]
    delta_a_star=0.01,     # Spin parameter difference
    num_points=100,        # Number of points
    save_path='3d_annihilation.png'
)
```

### Example 2: Single Transition Process

```python
from DetectorDistanceReach import plot_distance_reach

# Plot 5f→4f transition process
plot_distance_reach(
    h_det=1e-24,           # Detection threshold
    alpha=0.1,             # Fine structure constant
    process='5f 4f',       # Transition process
    bh_mass_range=(0.1, 50),
    a_star=0.9,            # Spin parameter
    save_path='5f_4f_transition.png'
)
```

### Example 3: Compare Multiple Processes

```python
from DetectorDistanceReach import plot_multiple_processes

# Compare several processes
processes = ['2p', '3p', '3d', '3p 2p']

results = plot_multiple_processes(
    h_det=1e-24,
    alpha=0.1,
    processes=processes,
    bh_mass_range=(0.1, 50),
    save_path='comparison.png'
)

# Access individual results
for process, (masses, distances) in results.items():
    print(f"{process}: max distance = {max(distances):.4e} kpc")
```

### Example 4: Calculate Specific Points

```python
from DetectorDistanceReach import calc_distance_reach_ann, calc_distance_reach_trans

# Annihilation process
dist_ann = calc_distance_reach_ann(
    h_det=1e-24,
    alpha=0.1,
    process='2p',
    bh_mass_solar=1.0,
    delta_a_star=0.01
)
print(f"2p annihilation reach: {dist_ann:.4e} kpc")

# Transition process
dist_trans = calc_distance_reach_trans(
    h_det=1e-24,
    alpha=0.1,
    process='3p 2p',
    bh_mass_solar=1.0,
    a_star=0.9
)
print(f"3p→2p transition reach: {dist_trans:.4e} kpc")
```

## Function Reference

### Main Plotting Functions

#### `plot_distance_reach(h_det, alpha, process, ...)`
Create a distance reach plot for a single process.

**Parameters:**
- `h_det` (float): Detection threshold strain (dimensionless)
- `alpha` (float): Fine structure constant (dimensionless)
- `process` (str): Process name (e.g., '2p', '3d', '5f 4f')
- `bh_mass_range` (tuple, optional): (min, max) BH mass in M_☉
- `delta_a_star` (float): Spin difference for annihilation (default: 0.01)
- `a_star` (float): Spin parameter for transitions (default: 0.9)
- `m_quantum` (int): Azimuthal quantum number (default: 1)
- `num_points` (int): Number of points to plot (default: 100)
- `save_path` (str, optional): Path to save figure
- `show_plot` (bool): Whether to display plot (default: True)

**Returns:**
- `(bh_masses, distances)`: Arrays of BH masses [M_☉] and distances [kpc]

#### `plot_multiple_processes(h_det, alpha, processes, ...)`
Plot multiple processes on the same figure.

**Parameters:**
- `h_det`, `alpha`, `bh_mass_range`, etc.: Same as above
- `processes` (list): List of process strings

**Returns:**
- `dict`: Dictionary mapping process names to (masses, distances) tuples

### Calculation Functions

#### `calc_distance_reach_ann(h_det, alpha, process, bh_mass_solar, ...)`
Calculate distance reach for a single annihilation process.

**Returns:** Distance in kpc

#### `calc_distance_reach_trans(h_det, alpha, process, bh_mass_solar, ...)`
Calculate distance reach for a single transition process.

**Returns:** Distance in kpc

### Utility Functions

#### `eV_inv_to_kpc(r_eV_inv)`
Convert distance from eV⁻¹ to kiloparsecs.

#### `parse_process(process)`
Parse process string to determine type and extract quantum numbers.

#### `extract_quantum_numbers(process)`
Extract n, l quantum numbers from process string.

## Available Processes

### Annihilation Processes
Available processes from `ConvertedFunctions.py`:
- `'2p'`, `'3p'`, `'3d'`, `'4p'`, `'4d'`, `'4f'`
- `'5p'`, `'5d'`, `'5f'`, `'5g'`
- `'6p'`, `'6d'`, `'6f'`, `'6g'`, `'6h'`

### Transition Processes
Format: `'n_e l_e n_g l_g'` (excited → ground)

Examples:
- `'3p 2p'`, `'4p 2p'`, `'4p 3p'`, `'4d 3d'`
- `'5p 4p'`, `'5d 4d'`, `'5f 4f'`
- `'6g 5g'`, `'7h 6h'`, `'8i 7i'`, `'9k 8k'`

And many more (see `ConvertedFunctions.py` for complete list).

## Parameter Guidelines

### Detection Threshold (`h_det`)
Typical values:
- LIGO/Virgo: ~10⁻²³ to 10⁻²⁴
- Advanced detectors: ~10⁻²⁵
- Future detectors: ~10⁻²⁶

### Fine Structure Constant (`alpha`)
- Typical range: 0.01 to 0.5
- Common values: 0.05, 0.1, 0.2
- Must satisfy validity conditions for superradiance

### Black Hole Mass Range
Suggested ranges:
- For α < 0.1: 0.1 to 100 M_☉
- For α < 0.3: 0.01 to 10 M_☉
- For α > 0.3: 0.001 to 1 M_☉

### Spin Parameters
- `delta_a_star` (annihilation): Typically 0.001 to 0.1
- `a_star` (transition): Typically 0.5 to 0.99

## Output

The module produces:
1. **Plots**: PNG files showing distance reach vs BH mass
2. **Data**: Arrays of BH masses and corresponding distances
3. **Console output**: Test results and validation messages

### Plot Features
- Logarithmic scales on both axes
- Labeled axes with units
- Title with parameter values
- Grid for readability
- Legend for multiple processes

## Testing

Run the built-in test suite:

```bash
python DetectorDistanceReach.py
```

Run extended tests:

```bash
python extended_tests.py
```

The test suites will:
- Verify all calculation functions
- Test annihilation processes
- Test transition processes
- Generate example plots
- Test parameter variations

## Physics Background

### Annihilation Process
Black holes with superradiant clouds can emit gravitational waves through annihilation:
- Strain: `h ∝ sqrt(Γ_a/(r² ω)) × n_max`
- Distance reach: `r_max = sqrt(8 G_N Γ_a/ω) × n_max / h_det`

### Transition Process
Transitions between energy levels also emit GWs:
- Strain: `h ∝ sqrt(Γ_sr²/(r² ω Γ_tr))`
- Distance reach: `r_max = sqrt(4 G_N Γ_sr²/(ω Γ_tr)) / h_det`

Where:
- `Γ_a`: Annihilation rate
- `Γ_tr`: Transition rate
- `Γ_sr`: Superradiance rate
- `ω`: Frequency
- `n_max`: Maximum occupation number
- `r`: Distance

## Notes

1. **Units**: All internal calculations use natural units (ℏ = c = 1). Outputs are converted to conventional units (kpc, M_☉).

2. **Validity**: Results are only valid when:
   - Superradiance condition is satisfied
   - α is not too large (typically α < 0.5)
   - Black hole mass is appropriate for the given α

3. **Accuracy**: Uses numerical integration (trapezoidal rule) for solid angle integrals with default 10,000 points.

4. **Performance**: Calculations are vectorized for efficiency. 100-point plots typically take a few seconds.

## References

Functions and formulas are based on:
- Arvanitaki & Dubovsky (2010) - Superradiance theory
- Arvanitaki et al. (2015) - Gravitational atom physics
- ConvertedFunctions.py - Differential power calculations
- ParamCalculator.py - Rate and parameter calculations

## Files in This Module

- `DetectorDistanceReach.py`: Main module with all functions
- `extended_tests.py`: Extended test suite
- `README.md`: This file
- `*.png`: Generated plot files

## Author

Created for MPhys project on gravitational wave detection from primordial black holes.

## License

For academic use.
