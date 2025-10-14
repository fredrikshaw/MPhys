import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.integrate import solve_ivp

from SuperradianceGrowthRate import calc_gamma, calc_alpha

np.seterr(all='ignore')  # Suppress all numpy warnings
warnings.filterwarnings('ignore')  # Suppress all Python warnings

SOLAR_MASS = 1.988e30  # [kg]

# ---------------------------------------------------------------
#  PHYSICS FUNCTIONS
# ---------------------------------------------------------------

def calc_omega_tr(n_g, n_e, mu_a, alpha):
    return 0.5 * mu_a * alpha**2 * (1/n_g**2 - 1/n_e**2)

def calc_transition_rate_APPROX(G_N, alpha, r_g):
    return 1e-7 * G_N * alpha**9 / r_g**3

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

def run_simulation(bh_mass_sm=10, bh_spin=0.9, alpha=1,
                   l_g=4, m_g=4, n_g=5,
                   l_e=4, m_e=4, n_e=6,
                   gamma_g_override=None, gamma_e_override=None,
                   transition_rate_override=1e-72,
                   distance_kpc=10, t_max_years=1e5, n_points=int(1e5)):
    """
    Run the simulation with logarithmic variable integration.

    bh_spin is a* not a
    
    """

    # --- Constants and conversions ---
    G_N = 6.708e-57  # eV^-2
    kpc_to_meters = 3.085677581e19
    meters_to_ev = 1 / 1.973269804e-7
    r = distance_kpc * kpc_to_meters * meters_to_ev

    # --- Convert BH mass ---
    m_bh_J = bh_mass_sm * SOLAR_MASS * constants.c ** 2  # [J]
    m_bh_ev = m_bh_J / constants.e                        # [eV]
    r_g = G_N * m_bh_ev                                   # [eV^-1]
    axion_mass = alpha / r_g

    inv_ev_to_years = 2.09e-23  # conversion factor

    a = bh_spin * r_g

    # --- Gamma values ---
    gamma_e_ev = calc_gamma(l=l_e, m=m_e, n=n_e, a=a, r_g=r_g, mu_a=axion_mass)
    gamma_g_ev = calc_gamma(l=l_g, m=m_g, n=n_g, a=a, r_g=r_g, mu_a=axion_mass)
    ## === Conversion of gamma into inverse years ===
    gamma_e = gamma_e_ev / inv_ev_to_years
    gamma_g = gamma_g_ev / inv_ev_to_years


    if gamma_e_override is not None and gamma_g_override is not None:
        gamma_e = gamma_e_override
        gamma_g = gamma_g_override

    omega_tr = calc_omega_tr(n_g=n_g, n_e=n_e, mu_a=axion_mass, alpha=alpha)
    transition_rate = transition_rate_override  # years^-1

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
        print(f"\n [WARNING] Integer Overflow: solve_ivp stopped early, N_g exceeded 1e100 at t ≈ {t_stop:.2f} years.\n")
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
            'l_g': l_g, 'm_g': m_g, 'n_g': n_g,
            'l_e': l_e, 'm_e': m_e, 'n_e': n_e,
            'gamma_g': gamma_g,
            'gamma_e': gamma_e,
             # 'gamma_g_calc': gamma_g_calc,
             # 'gamma_e_calc': gamma_e_calc,
            'transition_rate': transition_rate,
            'axion_mass': axion_mass,
            'omega_tr': omega_tr,
            'distance_kpc': distance_kpc
        }
    }

    return results

# ---------------------------------------------------------------
#  PLOTTING
# ---------------------------------------------------------------

def plot_results(results):
    times = results['times']
    num_g = results['num_g']
    num_e = results['num_e']
    h = results['h']

    # Detect collapse of N_e (minimum)
    collapse_index = np.argmin(num_e)
    collapse_time = times[collapse_index]
    timewindow = 100 # [years]
    xlim = (max(0, collapse_time - timewindow), collapse_time + timewindow)

    print(f"Excited-state population collapse at ~{collapse_time:.2f} years.")

    fig, ax1 = plt.subplots(figsize=(8, 10))
    line1 = ax1.plot(times, num_g, label=r'$N_g$ (5g)', color='blue', linestyle='dashed')
    line2 = ax1.plot(times, num_e, label=r'$N_e$ (6g)', color='orange')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$N$')
    ax1.set_xlabel("Time [years]")
    ax1.set_ylim(1e50, 1e75)
    # ax1.set_xlim(xlim)

    ax2 = ax1.twinx()
    line3 = ax2.plot(times, h, label=r'$h$', color='black', linestyle="dotted")
    ax2.set_yscale('log')
    ax2.set_ylabel(r'$h$')
    ax2.set_ylim(1e-35, 1e-23)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, framealpha=0)

    plt.tight_layout()
    filename = f"LevelTransitions_LogIntegration.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"\n [OUTPUT] Figure saved as '{filename}' in the current directory.\n")
    plt.show()

# ---------------------------------------------------------------
#  MAIN EXECUTION
# ---------------------------------------------------------------

if __name__ == "__main__":
    results = run_simulation(
        transition_rate_override=1e-72, 
        alpha=1, 
        bh_spin=0.9
    )
    plot_results(results)
