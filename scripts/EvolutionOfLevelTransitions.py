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

if __name__ == "__main__":
    bh_mass_sm = 10  # Solar masses
    bh_spin = 0.9
    alpha = 1
    G_N = 6.708e-57  # in eV^-2

    kpc_to_meters = 3.085677581e19  # 1 kpc in meters
    meters_to_ev = 1 / 1.973269804e-7  # Conversion factor from meters to eV^-1
    r = 10 * kpc_to_meters * meters_to_ev

    l_e = m_e = 4
    n_e = 6

    l_g = m_g = 4
    n_g = 5

    # --- Convert BH mass ---
    m_bh_J = bh_mass_sm * SOLAR_MASS * constants.c ** 2       # [J]
    m_bh_ev = m_bh_J / constants.e                               # [eV]
    G_N = 6.708e-57                                              # [eV^-2]
    r_g = G_N * m_bh_ev                                          # [eV^-1]
    r_g_SI = constants.G * bh_mass_sm * SOLAR_MASS / (constants.c)**2     # [m]

    axion_mass = alpha / r_g
    
    # Keep rates in natural units [eV] - no conversion needed
    inv_ev_to_years = 2.09e-23

    gamma_e_ev = -calc_gamma(l=l_e, m=m_e, n=n_e, a=bh_spin, r_g=r_g, mu_a=axion_mass)  # [eV]
    gamma_g_ev = -calc_gamma(l=l_g, m=m_g, n=n_g, a=bh_spin, r_g=r_g, mu_a=axion_mass)  # [eV]

    gamma_e = gamma_e_ev / inv_ev_to_years  # [years⁻¹]
    gamma_g = gamma_g_ev / inv_ev_to_years  # [years⁻¹]

    gamma_e = 0.1
    gamma_g = 0.08

    omega_tr = calc_omega_tr(n_g=n_g, n_e=n_e, mu_a=axion_mass, alpha=alpha)
    transition_rate_ev = calc_transition_rate_APPROX(G_N=G_N, alpha=alpha, r_g=r_g)  # [eV]
    transition_rate = 1e-72 #transition_rate_ev / inv_ev_to_years  # [years⁻¹]
    

    y0 = [1, 1]  # Initial populations for n_g and n_e
    
    # Define time array in years, then convert to natural units [eV^-1]
    times = np.linspace(0, 5e3, int(1e4))  # in years
    t_span = (0, times[-1])

    print("r_g (raw) =", r_g)
    print("axion_mass (mu_a) =", axion_mass)
    print("Gamma_e (eV) =", gamma_e)
    print("Gamma_g (eV) =", gamma_g)
    print("transition_rate (eV) =", transition_rate)
    print("Total simulated time [eV^-1] =", times[-1])
    dt = times[1]-times[0]
    print("dt [eV^-1] =", dt)
    print("Gamma_e * total_time =", gamma_e * times[-1])
    print("Gamma_e * dt (per step fractional change) =", gamma_e * dt)
    
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

    # extract results
    num_g = sol.y[0]
    num_e = sol.y[1]

    ## convert transition rate into eV again 
    transition_rate *= inv_ev_to_years 
    ## calculate GW wave strain 
    h = np.sqrt(4 * G_N / (r**2 * omega_tr) * num_g * num_e * transition_rate)

    fig, ax1 = plt.subplots(figsize=(8, 10))

    # Plot all curves
    line1 = ax1.plot(times, num_g, label=r'$N_g$ (5g)', color='blue', linestyle='dashed')
    line2 = ax1.plot(times, num_e, label=r'$N_e$ (6g)', color='orange')
    
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$N$')
    ax1.set_xlabel("Time [years]")
    ax1.set_ylim(1e50, 1e75)
    ax1.set_xlim(1500, 1800)
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
    plt.show()