import sys
from os.path import dirname, abspath, join
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.integrate import solve_ivp


# Add the "Results Pipeline" sub-folder to the Python path
current_dir = dirname(abspath(__file__))
results_pipeline_dir = join(current_dir, "Results Pipeline")
sys.path.append(results_pipeline_dir)

# Import the required functions from ParamCalculator
from ParamCalculator import (
    calc_superradiance_rate,
    calc_omega_transition,
    calc_transition_rate,
    calc_rg_from_bh_mass,
    G_N
)

np.seterr(all='ignore')  # Suppress all numpy warnings
warnings.filterwarnings('ignore')  # Suppress all Python warnings

SOLAR_MASS = 1.988e30  # [kg]



# ---------------------------------------------------------------
#  UTILITY FUNCTIONS
# ---------------------------------------------------------------

def quantum_numbers_to_spectroscopic(n, l):
    """
    Convert quantum numbers to spectroscopic notation (e.g., 5g, 6h).
    
    Args:
        n: principal quantum number
        l: orbital angular momentum quantum number
        
    Returns:
        String in spectroscopic notation (e.g., "5g")
    """
    # Spectroscopic notation letters for l values
    l_to_letter = {
        0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h',
        6: 'i', 7: 'k', 8: 'l', 9: 'm', 10: 'n', 11: 'o', 12: 'q'
    }
    
    if l not in l_to_letter:
        return f"{n}(l={l})"  # Fallback for l > 12
    
    return f"{n}{l_to_letter[l]}"

# ---------------------------------------------------------------
#  PHYSICS FUNCTIONS
# ---------------------------------------------------------------

def calc_omega_tr(n_g, n_e, mu_a, alpha):
    return 0.5 * mu_a * alpha**2 * (1/n_g**2 - 1/n_e**2)

# ---------------------------------------------------------------
#  LOGARITHMIC DIFFERENTIAL EQUATIONS
# ---------------------------------------------------------------

def rhs_log(t, y, gamma_g, gamma_e, transition_rate):
    """
    ODE system for ln(n_g) and ln(n_e) with overflow protection.
    """
    x, y_ = y
    exp_x = np.exp(np.clip(x, -700, 700))
    exp_y = np.exp(np.clip(y_, -700, 700))

    dx = gamma_g + transition_rate * exp_y
    dy = gamma_e - transition_rate * exp_x

    # Limit derivative magnitude to help solver stability
    dx = np.clip(dx, -1e5, 1e5)
    dy = np.clip(dy, -1e5, 1e5)

    return [dx, dy]

# ---------------------------------------------------------------
#  MAIN SIMULATION
# ---------------------------------------------------------------

def run_simulation(bh_mass_sm=1e-11, bh_spin=0.687, alpha=0.1,
                   transition="3p 2p",
                   gamma_g_override=None, gamma_e_override=None,
                   transition_rate_override=None,
                   distance_kpc=10, t_max_years=1e6, n_points=int(1e5),
                   swap_sr_rates=False):
    """
    Run the simulation with logarithmic variable integration.

    Args:
        bh_mass_sm: Black hole mass in solar masses
        bh_spin: a* (dimensionless spin parameter)
        alpha: fine structure constant
        transition: Transition specification like "3p 2p" or "6g 5g"
        gamma_g_override: Override for ground state SR rate (years^-1)
        gamma_e_override: Override for excited state SR rate (years^-1)
        transition_rate_override: Override for transition rate (years^-1)
        distance_kpc: Distance in kiloparsecs
        t_max_years: Maximum simulation time in years
        n_points: Number of time points
        swap_sr_rates: If True, swap gamma_e and gamma_g (for fake plot)
    
    """
    
    # Parse transition string (e.g., "3p 2p" or "6g 5g")
    parts = transition.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Transition must be in format 'ne_level ng_level' (e.g., '3p 2p'), got: {transition}")
    
    level_e_str = parts[0]
    level_g_str = parts[1]
    
    # Extract n and l from spectroscopic notation
    def parse_level(level_str):
        l_to_number = {
            's': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5,
            'i': 6, 'j': 7, 'k': 8, 'l': 9, 'm': 10, 'n': 11, 'o': 12, 'q': 13
        }
        n = int(level_str[:-1])
        l_letter = level_str[-1]
        l = l_to_number.get(l_letter)
        if l is None:
            raise ValueError(f"Unknown orbital letter: {l_letter}")
        return n, l
    
    n_e, l_e = parse_level(level_e_str)
    n_g, l_g = parse_level(level_g_str)
    
    # For superradiance with l=m constraint
    m_e = l_e
    m_g = l_g

    # --- Constants and conversions ---
    kpc_to_meters = 3.085677581e19
    meters_to_ev = 1 / 1.973269804e-7
    r = distance_kpc * kpc_to_meters * meters_to_ev

    # --- Calculate r_g using ParamCalculator ---
    r_g = calc_rg_from_bh_mass(bh_mass_sm)
    axion_mass = alpha / r_g  # [eV]

    inv_ev_to_years = 2.09e-23  # conversion factor
    ev_to_years = 31556926/6.582119569e-16  # conversion factor

    # Calculate gamma values using ParamCalculator
    gamma_e = calc_superradiance_rate(l=l_e, m=m_e, n=n_e, a_star=bh_spin, r_g=r_g, alpha=alpha)
    gamma_g = calc_superradiance_rate(l=l_g, m=m_g, n=n_g, a_star=bh_spin, r_g=r_g, alpha=alpha)

    gamma_e *= ev_to_years # Convert from eV to years^-1
    gamma_g *= ev_to_years # Convert from eV to years^-1

    # Swap if requested (for fake plot)
    if swap_sr_rates:
        gamma_e, gamma_g = gamma_g, gamma_e

    # === set override values if applicable ===
    if gamma_e_override is not None:
        gamma_e = gamma_e_override
    if gamma_g_override is not None:
        gamma_g = gamma_g_override

    # Calculate transition frequency using ParamCalculator
    omega_tr = calc_omega_transition(r_g, alpha, n_e, n_g)
    
    # --- Transition rate using ParamCalculator ----
    if transition_rate_override is None:
        # Get transition rate in eV
        transition_rate_ev = calc_transition_rate(
            transition=transition,
            alpha=alpha,
            omega=omega_tr,
            G_N=G_N,
            r_g=r_g
        )
        # Convert to years^-1
        transition_rate = transition_rate_ev * ev_to_years
    else:
        transition_rate = transition_rate_override


    # --- Time setup ---
    times = np.linspace(0, t_max_years, n_points)
    t_span = (0, times[-1])

    print("Simulation parameters:")
    print(f"BH mass: {bh_mass_sm} solar masses")
    print(f"BH spin: {bh_spin}")
    print(f"Alpha: {alpha}")
    print(f"Gamma_e = {gamma_e} years⁻¹")
    print(f"Gamma_g = {gamma_g} years⁻¹")
    print(f"transition_rate = {transition_rate} years⁻¹")
    print(f"Total simulated time = {times[-1]} years")

    # --- Event: stop if N_g > 1e100 ---
    def stop_if_ng_too_large(t, y, gamma_g, gamma_e, transition_rate):
        x, y_ = y
        N_g = np.exp(np.clip(x, -700, 700))
        return np.log10(N_g) - 100  # Trigger at N_g = 1e100
    stop_if_ng_too_large.terminal = True
    stop_if_ng_too_large.direction = 1

    # --- Initial conditions (log variables) ---
    n_g0, n_e0 = 1.0, 1.0
    y0_log = [np.log(n_g0), np.log(n_e0)]

    # --- Solve ODE system ---
    sol = solve_ivp(
        rhs_log,
        t_span,
        y0_log,
        args=(gamma_g, gamma_e, transition_rate),
        method='Radau',
        t_eval=times,
        rtol=1e-6,
        atol=1e-9,
        events=stop_if_ng_too_large,
    )

    # --- Use solver’s native arrays (prevents shape mismatch) ---
    times = sol.t
    num_g = np.exp(np.clip(sol.y[0], -700, 700))
    num_e = np.exp(np.clip(sol.y[1], -700, 700))

    # --- Check for early termination ---
    if sol.t_events[0].size > 0:
        t_stop = sol.t_events[0][0]
        print(f"\n[WARNING] Integer Overflow: solve_ivp stopped early, N_g exceeded 1e100 at t ≈ {t_stop:.2f} years.\n")
    else:
        t_stop = times[-1]

    # --- Gravitational-wave strain ---
    transition_rate_ev_calc = transition_rate * inv_ev_to_years
    h = np.sqrt(4 * G_N / (r**2 * omega_tr) * num_g * num_e * transition_rate_ev_calc)

    results = {
        'times': times,
        'num_g': num_g,
        'num_e': num_e,
        'h': h,
        'parameters': {
            'bh_mass_sm': bh_mass_sm,
            'bh_spin': bh_spin,
            'alpha': alpha,
            'transition': transition,
            'l_g': l_g, 'm_g': m_g, 'n_g': n_g,
            'l_e': l_e, 'm_e': m_e, 'n_e': n_e,
            'gamma_g': gamma_g,
            'gamma_e': gamma_e,
            'transition_rate': transition_rate,
            'axion_mass': axion_mass,
            'omega_tr': omega_tr,
            'distance_kpc': distance_kpc,
            'swap_sr_rates': swap_sr_rates
        }
    }

    return results

# ---------------------------------------------------------------
#  PLOTTING
# ---------------------------------------------------------------

# ---------------------------------------------------------------
#  PLOTTING
# ---------------------------------------------------------------

def plot_results(results, save_filename=None, h_ylim=None):
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

    times = results['times']
    num_g = results['num_g']
    num_e = results['num_e']
    h = results['h']
    params = results['parameters']

    # Detect collapse of N_e (minimum)
    collapse_index = np.argmin(num_e)
    collapse_time = times[collapse_index]
    timewindow = max(times) / 30  # [years]
    xlim = (max(0, collapse_time - timewindow), collapse_time + timewindow - 500)
    
    # Calculate h_ylim if not provided (from peak h value)
    if h_ylim is None:
        h_peak = np.max(h)
        h_ylim = (h_peak * 1e-8, h_peak * 10)

    # Print to terminal (no box)
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"Excited-state population collapse at ~{collapse_time:.2f} years.")
    
    # Get spectroscopic notation for transitions
    level_e = quantum_numbers_to_spectroscopic(params['n_e'], params['l_e'])
    level_g = quantum_numbers_to_spectroscopic(params['n_g'], params['l_g'])
    print(f"\nTransition: {level_e} → {level_g}")
    print(f"BH mass: {params['bh_mass_sm']} M☉")
    print(f"BH spin: {params['bh_spin']}")
    print(f"Alpha: {params['alpha']}")
    print(f"Axion mass: {params['axion_mass']:.2e} eV")
    print(f"Distance: {params['distance_kpc']} kpc")
    
    # Compare Gamma_e and Gamma_g
    gamma_e = params.get('gamma_e')
    gamma_g = params.get('gamma_g')
    swap_sr_rates = params.get('swap_sr_rates', False)
    
    if gamma_e is not None and gamma_g is not None:
        print(f"\nSuperradiance rates:")
        print(f"  Gamma_e ({level_e}): {gamma_e:.2e} years⁻¹")
        print(f"  Gamma_g ({level_g}): {gamma_g:.2e} years⁻¹")
        
        if gamma_e > gamma_g:
            comparison_text = "Gamma_e > Gamma_g"
        elif gamma_e < gamma_g:
            comparison_text = "Gamma_e < Gamma_g"
        else:
            comparison_text = "Gamma_e = Gamma_g"
        print(f"\nComparison: {comparison_text}")
        print(f"Rates swapped: {swap_sr_rates}")
    else:
        print("\nWarning: Gamma values missing from parameters")
    
    print(f"\nTransition rate: {params['transition_rate']:.2e} years⁻¹")
    print(f"Transition frequency: {params['omega_tr']:.2e} eV")
    print("="*60 + "\n")

    fig, ax1 = plt.subplots(figsize=(3, 4))
    
    # Plot with clear labels for legend
    line1 = ax1.plot(times, num_g, label=r'$N_g$', color='blue')
    line2 = ax1.plot(times, num_e, label=r'$N_e$', color='orange')
    
    # Create second y-axis for strain h
    ax2 = ax1.twinx()
    line3 = ax2.plot(times, h, label=r'$h$', color='black', alpha=0.7, linestyle="dashed")
    
    # Set scales and labels
    ax1.set_yscale('log')
    ax1.set_ylabel(r'Occupation Number $N$')
    ax1.set_xlabel("Time [years]")
    # ax1.set_ylim(1e25, 1e45)
    # ax1.set_xlim(xlim)
    ax1.grid()
    
    ax2.set_ylabel(r'Strain $h$', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_yscale('log')
    # ax2.set_ylim(h_ylim)
    for label in ax2.get_yticklabels()[::2]:
        label.set_visible(False)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
               loc='best', frameon=False)
    
    # Add comparison text as annotation on plot with LaTeX formatting
    if gamma_e is not None and gamma_g is not None:
        if gamma_e > gamma_g:
            comp_text = r'$\Gamma_e^{\text{sr}} > \Gamma_g^{\text{sr}}$'
        elif gamma_e < gamma_g:
            comp_text = r'$\Gamma_g^{\text{sr}} > \Gamma_e^{\text{sr}}$'
        else:
            comp_text = r'$\Gamma_e^{\text{sr}} = \Gamma_g^{\text{sr}}$'
        
        # Place text in upper left corner
        ax1.text(0.02, 0.98, comp_text,
                transform=ax1.transAxes,
                fontsize=14,
                verticalalignment='top')
    
    # Use provided filename or construct one
    if save_filename is None:
        # Extract parameters for the file name
        alpha = params.get('alpha', 'unknown')
        bh_mass = params.get('bh_mass_sm', 'unknown')
        spin = params.get('bh_spin', 'unknown')
        transition = params.get('transition', 'unknown').replace(' ', '_')
        swap_label = "_swapped" if swap_sr_rates else "_original"

        # Construct the file name dynamically
        file_name = f"plot_{transition}_alpha={alpha}_bhmass={bh_mass}_spin={spin}{swap_label}.pdf"

        # Replace invalid characters in the file name (e.g., '/', ':', etc.)
        import re
        file_name = re.sub(r'[^\w\-.]', '_', file_name)
        
        # Save to Final plots folder
        import os
        final_plots_dir = join(current_dir, "Results Pipeline", "Final plots")
        os.makedirs(final_plots_dir, exist_ok=True)
        file_name = join(final_plots_dir, file_name)
    else:
        file_name = save_filename

    # Save the plot with the dynamic file name
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)  # Save the figure with high resolution
    print(f"Plot saved as: {file_name}")

    # Show the plot
    plt.show()


# ---------------------------------------------------------------
#  MAIN EXECUTION
# ---------------------------------------------------------------


if __name__ == "__main__":
    # Run simulation with original SR rates to determine h_ylim
    results_original = run_simulation(
        alpha=0.1,
        bh_spin=0.687,
        bh_mass_sm=1e-11,
        transition="6g 5g",
        swap_sr_rates=False
    )
    
    # Determine which case has gamma_e > gamma_g
    gamma_e_orig = results_original['parameters']['gamma_e']
    gamma_g_orig = results_original['parameters']['gamma_g']
    
    # Calculate h_ylim based on the gamma_e > gamma_g case
    if gamma_e_orig > gamma_g_orig:
        # Original case has gamma_e > gamma_g, use it for h_ylim
        h_peak = np.max(results_original['h'])
    else:
        # Swapped case will have gamma_e > gamma_g, run it first
        results_temp = run_simulation(
            alpha=0.1,
            bh_spin=0.687,
            bh_mass_sm=1e-11,
            transition="6g 5g",
            swap_sr_rates=True
        )
        h_peak = np.max(results_temp['h'])
    
    # Set h_ylim: 8 orders of magnitude below peak to 1 order above peak
    h_ylim = (h_peak * 1e-8, h_peak * 10)
    
    # Plot original results
    plot_results(results_original, h_ylim=h_ylim)
    
    # Run simulation with swapped SR rates
    results_swapped = run_simulation(
        alpha=0.1,
        bh_spin=0.687,
        bh_mass_sm=1e-11,
        transition="6g 5g",
        swap_sr_rates=True
    )
    plot_results(results_swapped, h_ylim=h_ylim)
