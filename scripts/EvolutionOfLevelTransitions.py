import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.integrate import solve_ivp
from tqdm import tqdm

from SuperradianceGrowthRate import calc_gamma, calc_alpha

np.seterr(all='ignore')  # Suppress all numpy warnings
warnings.filterwarnings('ignore')  # Suppress all Python warnings

SOLAR_MASS = 1.988e30  # [kg]

def calc_omega_tr(n_g, n_e, mu_a, alpha):
    # Transition frequency
    return 0.5 * mu_a * alpha**2 * (1/n_g**2 - 1/n_e**2)

def calc_transition_rate_APPROX(G_N, alpha, r_g):
    return 1e-7 * G_N * alpha**9 / r_g**3

def dn_g_dt(sr_rate_g, n_g, n_e, transition_rate):
    return sr_rate_g * n_g + transition_rate * n_g * n_e

def dn_e_dt(sr_rate_e, n_g, n_e, transition_rate):
    return sr_rate_e * n_e - transition_rate * n_g * n_e

def h_tr(G_N, r, omega_tr, n_g, n_e, transition_rate):
    return np.sqrt(4*G_N/(r**2 * omega_tr) * n_g * n_e * transition_rate)

def rhs(t, y, gamma_g, gamma_e, transition_rate):
    n_g, n_e = y
    dn_g = gamma_g * n_g + transition_rate * n_g * n_e
    dn_e = gamma_e * n_e - transition_rate * n_g * n_e
    return [dn_g, dn_e]

def run_simulation(bh_mass_sm=10, bh_spin=0.9, alpha=1, 
                   l_g=4, m_g=4, n_g=5, 
                   l_e=4, m_e=4, n_e=6,
                   gamma_g_override=0.08, gamma_e_override=0.1,
                   transition_rate_override=1e-72,
                   distance_kpc=10, t_max_years=5e3, n_points=int(1e4)):
    """
    Run the simulation with specified parameters.
    
    Parameters:
    -----------
    bh_mass_sm : float
        Black hole mass in solar masses
    bh_spin : float
        Black hole spin (0-1)
    alpha : float
        Coupling constant
    l_g, m_g, n_g : int
        Quantum numbers for ground state
    l_e, m_e, n_e : int
        Quantum numbers for excited state
    gamma_g_override, gamma_e_override : float
        Override calculated gamma values with these (in years⁻¹)
    transition_rate_override : float
        Override calculated transition rate with this value
    distance_kpc : float
        Distance to source in kiloparsecs
    t_max_years : float
        Maximum simulation time in years
    n_points : int
        Number of time points
    
    Returns:
    --------
    dict : Dictionary containing all results and parameters
    """
    
    G_N = 6.708e-57  # in eV^-2

    # Distance conversion
    kpc_to_meters = 3.085677581e19  # 1 kpc in meters
    meters_to_ev = 1 / 1.973269804e-7  # Conversion factor from meters to eV^-1
    r = distance_kpc * kpc_to_meters * meters_to_ev

    # --- Convert BH mass ---
    m_bh_J = bh_mass_sm * SOLAR_MASS * constants.c ** 2       # [J]
    m_bh_ev = m_bh_J / constants.e                               # [eV]
    G_N = 6.708e-57                                              # [eV^-2]
    r_g = G_N * m_bh_ev                                          # [eV^-1]
    r_g_SI = constants.G * bh_mass_sm * SOLAR_MASS / (constants.c)**2     # [m]

    axion_mass = alpha / r_g
    
    # Conversion factor
    inv_ev_to_years = 2.09e-23

    # Calculate gamma values (but allow override)
    gamma_e_ev = -calc_gamma(l=l_e, m=m_e, n=n_e, a=bh_spin, r_g=r_g, mu_a=axion_mass)  # [eV]
    gamma_g_ev = -calc_gamma(l=l_g, m=m_g, n=n_g, a=bh_spin, r_g=r_g, mu_a=axion_mass)  # [eV]

    gamma_e_calc = gamma_e_ev / inv_ev_to_years  # [years⁻¹]
    gamma_g_calc = gamma_g_ev / inv_ev_to_years  # [years⁻¹]

    # Use overrides if provided
    gamma_e = gamma_e_override
    gamma_g = gamma_g_override

    omega_tr = calc_omega_tr(n_g=n_g, n_e=n_e, mu_a=axion_mass, alpha=alpha)
    transition_rate_ev = calc_transition_rate_APPROX(G_N=G_N, alpha=alpha, r_g=r_g)  # [eV]
    transition_rate = transition_rate_override  # [years⁻¹]

    y0 = [1, 1]  # Initial populations for n_g and n_e
    
    # Define time array
    times = np.linspace(0, t_max_years, n_points)  # in years
    t_span = (0, times[-1])

    print("Simulation parameters:")
    print(f"BH mass: {bh_mass_sm} solar masses")
    print(f"BH spin: {bh_spin}")
    print(f"Alpha: {alpha}")
    print(f"Ground state: n={n_g}, l={l_g}, m={m_g}")
    print(f"Excited state: n={n_e}, l={l_e}, m={m_e}")
    print(f"r_g (raw) = {r_g}")
    print(f"axion_mass (mu_a) = {axion_mass}")
    print(f"Gamma_e = {gamma_e} years⁻¹")
    print(f"Gamma_g = {gamma_g} years⁻¹")
    print(f"transition_rate = {transition_rate} years⁻¹")
    print(f"Total simulated time = {times[-1]} years")

    # Run integration
    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        args=(gamma_g, gamma_e, transition_rate),
        method='Radau',
        t_eval=times,
        rtol=1e-6,
        atol=1e-9,
    )

    # Extract results
    num_g = sol.y[0]
    num_e = sol.y[1]

    # Convert transition rate into eV for strain calculation
    transition_rate_ev_calc = transition_rate * inv_ev_to_years 
    
    # Calculate GW strain
    h = np.sqrt(4 * G_N / (r**2 * omega_tr) * num_g * num_e * transition_rate_ev_calc)

    # Return all results and parameters
    results = {
        'times': times,
        'num_g': num_g,
        'num_e': num_e,
        'h': h,
        'parameters': {
            'bh_mass_sm': bh_mass_sm,
            'bh_spin': bh_spin,
            'alpha': alpha,
            'l_g': l_g, 'm_g': m_g, 'n_g': n_g,
            'l_e': l_e, 'm_e': m_e, 'n_e': n_e,
            'gamma_g': gamma_g,
            'gamma_e': gamma_e,
            'gamma_g_calc': gamma_g_calc,
            'gamma_e_calc': gamma_e_calc,
            'transition_rate': transition_rate,
            'axion_mass': axion_mass,
            'omega_tr': omega_tr,
            'distance_kpc': distance_kpc
        }
    }
    
    return results

def plot_results(results, xlim=(1500, 1800)):
    """
    Plot the simulation results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_simulation
    xlim : tuple
        x-axis limits for the plot
    """
    times = results['times']
    num_g = results['num_g']
    num_e = results['num_e']
    h = results['h']
    
    fig, ax1 = plt.subplots(figsize=(8, 10))

    # Plot all curves
    line1 = ax1.plot(times, num_g, label=r'$N_g$ (5g)', color='blue', linestyle='dashed')
    line2 = ax1.plot(times, num_e, label=r'$N_e$ (6g)', color='orange')
    
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$N$')
    ax1.set_xlabel("Time [years]")
    ax1.set_ylim(1e50, 1e75)
    ax1.set_xlim(xlim[0], xlim[1])
    ax1.grid()

    # Create second y-axis for strain
    ax2 = ax1.twinx()
    line3 = ax2.plot(times, h, label=r'h', color='black', linestyle="dotted")
    
    ax2.set_yscale('log')
    ax2.set_ylabel(r'$r$')
    ax2.set_ylim(1e-35, 1e-23)

    # Combine all lines for a single legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    
    # Create single legend with transparent background
    ax1.legend(lines, labels, framealpha=0)  # framealpha=0 makes background transparent

    plt.tight_layout()
    # --- Save the figure ---
    filename = f"LevelTransitions.png"
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"\n✅ Figure saved as '{filename}' in the current directory.\n")


    plt.show()

if __name__ == "__main__":
    # Run simulation with default parameters
    results = run_simulation()
    
    # Plot results
    plot_results(results)
    
    # Example of running with different parameters:
    # results2 = run_simulation(
    #     bh_mass_sm=5, 
    #     bh_spin=0.7, 
    #     alpha=0.5,
    #     n_g=4, 
    #     n_e=5,
    #     gamma_g_override=0.05,
    #     gamma_e_override=0.08
    # )
    # plot_results(results2)