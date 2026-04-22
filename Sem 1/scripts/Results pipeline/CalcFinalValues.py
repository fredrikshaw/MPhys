from ParamCalculator import *

if __name__ == "__main__":
    # Constants
    G_N = 6.708e-57  # [eV^-2]

    # The only parameters that actually effect the GWs that get emitted
    omega_GHz = 1e-7  # [GHz] - used to calculate omega when alpha_override is None
    omega = omega_GHz * 4.135667696e-6  # [eV] - conversion: 1 GHz = 4.135667696e-6 eV
    alpha = 0.1    # unitless

    # Measured parameters
    astar_init = 0.9  # unitless  A value of 0.687 can be found from {https://arxiv.org/pdf/0712.3541}{Final spin from the coalescence of two black holes, Rezzola et. al.}
    h_det = 1e-24   # Unitless

    # Merger rate calculation params
    f_pbh = 1
    f_m1 = f_m2 = 1
    f_sup = 1

    # Params for calculating a strain at a given distance
    r_kpc = 10     # [kpc]
    r_m = r_kpc * 3.086e19   # [m] - conversion: 1 kpc = 3.086e16 m
    r = r_m / (1.9732705e-7)   # [eV^-1] - conversion: 1 m = 5.067731e6 eV^-1 (from ℏc)

    # Event definition
    gw_source = "annihilation"
    levels = "2p"
    m_lookup = {
        "s": 0,
        "p": 1,
        "d": 2,
        "f": 3,
        "g": 4,
        "h": 5,
        "i": 6,
        "k": 7,
        "l": 8,
        "m": 9,
        "n": 10,
        "o": 11,
        "q": 12,
        "r": 13,
        "t": 14,
    }

    if gw_source == "transition":
        n_e = int(levels[0])
        n_g = int(levels[3])
        m_e = l_e = m_lookup[levels[1]]
        m_g = l_g = m_lookup[levels[4]]
        r_g = calc_rg_from_omega_trans(omega, alpha, n_g, n_e)
        bh_mass = calc_bh_mass(r_g)
        delta_astar = calc_delta_astar(astar_init, r_g, alpha, n_e, m_e)   # unitless
        tran_rate = calc_transition_rate(levels, alpha, omega, G_N,r_g)    # [eV]
        pbh_merger_rate_gpc_yr = calc_merger_rate(f_sup, f_pbh, r_g, r_g, f_m1, f_m2) * 3e5  # [Gpc^-3 yr^-1]
        pbh_merger_rate_m_s = pbh_merger_rate_gpc_yr / ((3.086e25)**3 * 31556926)  # [m^-3 s^-1]
        pbh_merger_rate = pbh_merger_rate_m_s * ((1.9732705e-7)**3 * 6.582119569e-16)  # [eV^4]
        sr_rate = calc_superradiance_rate(l_e, m_e, n_e, astar_init, r_g, alpha)

        event_rate = calc_event_rate_tran(h_det, G_N, tran_rate, sr_rate, omega, pbh_merger_rate)
    elif gw_source == "annihilation":
        n = int(levels[0])
        m = l = m_lookup[levels[1]]
        r_g = calc_rg_from_omega_ann(omega, alpha, n)   # eV^-1
        bh_mass_solar = calc_bh_mass(r_g)   # M_solar
        bh_mass = r_g / G_N   # [eV] - calculate mass directly from r_g
        delta_astar = calc_delta_astar(astar_init, r_g, alpha, n, m)   # unitless
        ann_rate = calc_annihilation_rate(levels, alpha, omega, G_N, r_g)    # [eV]
        n_max = calc_n_max(bh_mass, delta_astar, m)
        detectable_radius = calc_detectable_radius_ann(h_det, ann_rate, omega, n_max)
        sr_rate = calc_superradiance_rate(l, m, n, astar_init, r_g, alpha)
        # Here we multiply by 3e5 to account for galactic density vs universe density
        pbh_merger_rate_gpc_yr = calc_merger_rate(f_sup, f_pbh, r_g, r_g, f_m1, f_m2) * 3e5  # [Gpc^-3 yr^-1]
        pbh_merger_rate_m_s = pbh_merger_rate_gpc_yr / ((3.086e25)**3 * 31556926)  # [m^-3 s^-1]
        # Convert to natural units: [eV^4]
        # Length: 1 m = 5.067731e6 eV^-1, so 1 m^-1 = (5.067731e6)^-1 eV = 1.973e-7 eV
        # Therefore: 1 m^-3 = (1.973e-7)^3 eV^3
        # Time: 1 s^-1 = ℏ eV where ℏ = 6.582119569e-16 eV·s
        pbh_merger_rate = pbh_merger_rate_m_s * ((1.9732705e-7)**3 * 6.582119569e-16)  # [eV^4]
        
        event_rate = calc_event_rate_ann(r_g, delta_astar, m, h_det, G_N, ann_rate, omega, pbh_merger_rate)  # [eV^-1]

    print("Black hole mass [M_solar]: ", bh_mass_solar)
    
    # Check superradiance condition
    omega_nlm = alpha**2 / (2 * n**2)  # Bohr energy in natural units (M_BH = 1)
    omega_H = astar_init / (2 * r_g)  # Horizon frequency
    astar_crit = 2 * r_g * m * omega_nlm  # Critical spin for superradiance
    
    print(f"Alpha: {alpha}")
    print(f"Initial spin (astar): {astar_init}")
    print(f"Critical spin (astar_crit): {astar_crit}")
    print(f"Superradiance condition (astar > astar_crit): {astar_init > astar_crit}")
    print(f"Omega_nlm: {omega_nlm}")
    print(f"m * Omega_H: {m * omega_H}")
    
    print("Superradiance rate [eV]: ", sr_rate)
    print("Superradiance rate [yr^-1]: ", sr_rate * 31556926/6.582119569e-16)
    print("Detectable radius [eV^-1]: ", detectable_radius)
    print("Detectable radius [m]: ", detectable_radius * 1.9733e-7)
    print("Detectable radius [kpc]: ", detectable_radius * 1.9733e-7 / 3.086e19)

    print("Events per second: ", event_rate/6.582119569e-16)  # ℏ = 6.582119569e-16 eV·s
    print("Events per year: ", event_rate * 31556926/6.582119569e-16)  # convert eV to years^-1
        
