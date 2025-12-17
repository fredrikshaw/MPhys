from ConvertedFunctions import diff_power_ann_dict, diff_power_trans_dict
import numpy as np
from scipy import constants


# This is a file containing all definitions needed to calculate
# various values related to the superradiant growth and
# consequent gravitational wave emission of primordial black
# holes. Where references are not given, it is likely that
# functions can be credited to {https://arxiv.org/pdf/1411.2263}
# {Discovering the QCD axion with black holes and gravitational
#  waves - Arvanitaki}.

G_N = 6.708e-57  # [eV^-2]

# ------------------------- integrate -----------------------------
def solid_angle_integral_from_dPdOmega(dPdtheta_func, Ntheta=10000):
    """
    Given a function dP/dOmega(theta) (independent of phi), compute
    the total power P = ∫ dΩ dP/dΩ = 2π ∫_0^π sinθ dθ dP/dΩ(θ).
    """
    theta = np.linspace(0.0, np.pi, Ntheta)
    dP_dOmega = dPdtheta_func(theta)
    integrand = np.sin(theta) * dP_dOmega  # integrand for polar integral
    P = 2 * np.pi * np.trapz(integrand, theta)
    return P


# ------------------------- alpha, omega, mu_a calculations -----------------------------
'''
So basically mu_a, r_g, alpha, omega and black hole mass are all linked together and can be solved from each other.
The goal is to be able to give as few as possible to solve for all other values
'''
def calc_rg_from_bh_mass(black_hole_mass_solar_mass):
    """
    Calculate gravitational radius from black hole mass.
    
    Args:
        black_hole_mass_solar_mass (float): Black hole mass [M_☉]
    
    Returns:
        float: Gravitational radius [eV^-1]
    """
    solar_mass = 1.988e30  # [kg]
    m_bh_J = black_hole_mass_solar_mass * solar_mass * constants.c ** 2       # [J]
    m_bh_ev = m_bh_J / constants.e                               # [eV]
    r_g = G_N * m_bh_ev                                          # [eV^-1]
    return r_g

def calc_bh_mass(r_g):
    """
    Calculate black hole mass from gravitational radius.
    
    Args:
        r_g (float): Gravitational radius [eV^-1]
    
    Returns:
        float: Black hole mass [M_☉]
    """
    solar_mass = 1.988e30  # [kg]
    m_bh_ev = r_g/G_N  # [eV]
    m_bh_J = m_bh_ev * constants.e  # [J]
    bh_mass = m_bh_J/(solar_mass * constants.c**2)  # [M_☉]
    return bh_mass

def calc_mu_a(alpha, r_g):
    """
    Calculate axion mass from fine structure constant and gravitational radius.
    
    Args:
        alpha (float): Fine structure constant (dimensionless)
        r_g (float): Gravitational radius [eV^-1]
    
    Returns:
        float: Axion mass [eV]
    """
    return alpha/r_g

def calc_omega_ann(r_g, alpha, n):
    """
    Compute annihilation frequency omega_ann for level n.
    
    Args:
        r_g (float): Gravitational radius [eV^-1]
        alpha (float): Fine structure constant (dimensionless)
        n (int): Principal quantum number (dimensionless)
    
    Returns:
        float: Annihilation frequency [eV]
    
    Formula:
        omega_ann = 2 * mu_a * (1 - alpha^2/(2 n^2))
        where mu_a = alpha/r_g [eV]
    """
    return 2.0 * (alpha/r_g) * (1.0 - alpha**2 / (2.0 * n**2))

def calc_omega_transition(r_g, alpha, n_e, n_g):
    """
    Compute transition frequency omega_tr between levels n_e and n_g.
    
    Args:
        r_g (float): Gravitational radius [eV^-1]
        alpha (float): Fine structure constant (dimensionless)
        n_e (int): Excited state principal quantum number (dimensionless)
        n_g (int): Ground state principal quantum number (dimensionless)
    
    Returns:
        float: Transition frequency [eV]
    
    Formula:
        omega_tr = mu_a * alpha^2/2 * (1/n_g^2 - 1/n_e^2)
        where mu_a = alpha/r_g [eV]
    """
    return 0.5 * (alpha/r_g) * alpha**2 * (1.0 / n_g**2 - 1.0 / n_e**2)


def calc_rg_from_omega_ann(omega, alpha, n):
    """
    Calculate gravitational radius from annihilation frequency (inverse calculation).
    
    Args:
        omega (float): Annihilation frequency [eV]
        alpha (float): Fine structure constant (dimensionless)
        n (int): Principal quantum number (dimensionless)
    
    Returns:
        float: Gravitational radius [eV^-1]
    """
    return 2.0 * (alpha/omega) * (1.0 - alpha**2 / (2.0 * n**2))


def calc_rg_from_omega_trans(omega, alpha, n_g, n_e):
    """
    Calculate gravitational radius from transition frequency (inverse calculation).
    
    Args:
        omega (float): Transition frequency [eV]
        alpha (float): Fine structure constant (dimensionless)
        n_g (int): Ground state principal quantum number (dimensionless)
        n_e (int): Excited state principal quantum number (dimensionless)
    
    Returns:
        float: Gravitational radius [eV^-1]
    """
    return 0.5 * (alpha/omega) * alpha**2 * (1.0 / n_g**2 - 1.0 / n_e**2)
    

def calc_alpha_ann(omega_ann, r_g, n, tol_imag=1e-8):
    """
    Solve for alpha from annihilation frequency (inverse calculation).
    
    Parameters:
    -----------
    omega_ann : float
        Annihilation frequency [eV]
    r_g : float
        Gravitational radius [eV^-1]
    n : int
        Principal quantum number (dimensionless)
    tol_imag : float, optional
        Tolerance for imaginary part of roots (default: 1e-8)
    
    Returns:
    --------
    alpha : float
        Fine structure constant (dimensionless)
    
    Notes:
    ------
    Solves the cubic equation:
        alpha^3 - 2 n^2 alpha + n^2 * (omega_ann * r_g) = 0
    Returns the smallest physically relevant real, positive root.
    """
    # polynomial: alpha^3 + 0*alpha^2 - 2 n^2 * alpha + n^2 * omega_ann * r_g = 0
    coeffs = [1.0, 0.0, -2.0 * (n**2), (n**2) * (omega_ann * r_g)]
    roots = np.roots(coeffs)

    # keep roots with (near-)zero imaginary part
    real_candidates = []
    for rt in roots:
        if abs(np.imag(rt)) <= tol_imag:
            re = float(np.real(rt))
            if re > 0.0:
                real_candidates.append(re)

    if not real_candidates:
        raise ValueError("No positive real root found for alpha (check inputs).")

    # choose the smallest positive real root (physically alpha is typically small)
    alpha = min(real_candidates)
    return alpha


def calc_alpha_trans(omega_tr, r_g, n_e, n_g):
    """
    Solve for alpha from transition frequency (inverse calculation).
    
    Args:
        omega_tr (float): Transition frequency [eV]
        r_g (float): Gravitational radius [eV^-1]
        n_e (int): Excited state principal quantum number (dimensionless)
        n_g (int): Ground state principal quantum number (dimensionless)
    
    Returns:
        float: Fine structure constant (dimensionless)
    
    Notes:
        Solves: alpha^3 = (2 * omega_tr * r_g) / Delta
        where Delta = 1/n_g^2 - 1/n_e^2
        Physically, Delta > 0 when n_g < n_e.
    """
    Delta = (1.0 / (n_g**2)) - (1.0 / (n_e**2))
    if Delta == 0.0:
        raise ValueError("Delta = 0 (n_g and n_e produce identical 1/n^2); cannot solve.")

    val = (2.0 * omega_tr * r_g) / Delta
    # real cube root that handles negative arguments
    alpha = np.sign(val) * (abs(val) ** (1.0 / 3.0))
    if alpha < 0.0:
        # physical alpha should be positive; if negative due to input signs, raise or take abs
        raise ValueError("Computed alpha is negative. Check sign of inputs (Delta, omega_tr, r_g).")
    return alpha


# ------------------------- integrate & rates -----------------------------
def calc_total_power_ann(level, alpha, G_N=1.0, r_g=1.0):
    """
    Calculate total annihilation power by integrating over solid angle.
    
    Args:
        level (str): Orbital level (e.g., '2p', '3d', '4f')
        alpha (float): Fine structure constant (dimensionless)
        G_N (float, optional): Gravitational constant [eV^-2]. Defaults to 1.0.
        r_g (float, optional): Gravitational radius [eV^-1]. Defaults to 1.0.
    
    Returns:
        float: Total annihilation power [eV^2]
    """
    func = lambda th: diff_power_ann_dict[level](alpha, th, G_N=G_N, r_g=r_g)
    return solid_angle_integral_from_dPdOmega(func)

def calc_total_power_trans(transition, alpha, G_N=1.0, r_g=1.0):
    """
    Calculate total transition power by integrating over solid angle.
    
    Args:
        transition (str): Transition specification (e.g., '6g 5g', '7h 6h', '5f 4f')
        alpha (float): Fine structure constant (dimensionless)
        G_N (float, optional): Gravitational constant [eV^-2]. Defaults to 1.0.
        r_g (float, optional): Gravitational radius [eV^-1]. Defaults to 1.0.
    
    Returns:
        float: Total transition power [eV^2]
    """
    func = lambda th: diff_power_trans_dict[transition](alpha, th, G_N=G_N, r_g=r_g)
    return solid_angle_integral_from_dPdOmega(func)

def calc_annihilation_rate(level, alpha, omega, G_N=1.0, r_g=1.0):
    """
    Compute annihilation rate Gamma_a per eq. (A12).
    
    Args:
        level (str): Orbital level (e.g., '2p', '3d', '4f')
        alpha (float): Fine structure constant (dimensionless)
        omega (float): Annihilation frequency [eV]
        G_N (float, optional): Gravitational constant [eV^-2]. Defaults to 1.0.
        r_g (float, optional): Gravitational radius [eV^-1]. Defaults to 1.0.
    
    Returns:
        float: Annihilation rate [eV] (inverse time in natural units)
    
    Formula:
        Gamma_a = (1/(2 * omega)) * ∫ dΩ (dP/dΩ)|ann
    """
    P = calc_total_power_ann(level, alpha, G_N=G_N, r_g=r_g)
    Gamma_a = P / (2.0 * omega)
    return Gamma_a

def calc_transition_rate(transition, alpha, omega, G_N=1.0, r_g=1.0):
    """
    Compute transition rate Gamma_t per eq. (A12).
    
    Args:
        transition (str): Transition specification (e.g., '6g 5g', '7h 6h', '5f 4f')
        alpha (float): Fine structure constant (dimensionless)
        omega (float): Transition frequency [eV]
        G_N (float, optional): Gravitational constant [eV^-2]. Defaults to 1.0.
        r_g (float, optional): Gravitational radius [eV^-1]. Defaults to 1.0.
    
    Returns:
        float: Transition rate [eV] (inverse time in natural units)
    
    Formula:
        Gamma_t = (1/omega) * ∫ dΩ (dP/dΩ)|tr
    """
    P = calc_total_power_trans(transition, alpha, G_N=G_N, r_g=r_g)
    Gamma_t = P / (omega)
    return Gamma_t

# ----------------- h peak calculations ------------------------

def calc_n_max(bh_mass, delta_a_star, m_quantum_number):
    """
    Calculate maximum occupation number for superradiance.
    
    Args:
        bh_mass (float): Black hole mass [eV]
        delta_a_star (float): Spin parameter difference (dimensionless)
        m_quantum_number (int): Azimuthal quantum number (dimensionless)
    
    Returns:
        float: Maximum occupation number (dimensionless)
    """
    return G_N * bh_mass**2 * delta_a_star / m_quantum_number

def calc_h_peak_ann(ann_rate, omega_ann, r, n_max):
    """
    Calculate peak gravitational wave strain from annihilation.
    
    Args:
        ann_rate (float): Annihilation rate [eV]
        omega_ann (float): Annihilation frequency [eV]
        r (float): Distance to source [eV^-1]
        n_max (float): Maximum occupation number (dimensionless)
    
    Returns:
        float: Peak strain amplitude (dimensionless)
    """
    root = np.sqrt(8*G_N*ann_rate / (r**2*omega_ann))
    return root * n_max

def calc_h_peak_trans(trans_rate, omega_trans, r, sr_rate):
    """
    Calculate peak gravitational wave strain from transition.
    
    Args:
        trans_rate (float): Transition rate [eV]
        omega_trans (float): Transition frequency [eV]
        r (float): Distance to source [eV^-1]
        sr_rate (float): Superradiance rate [eV]
    
    Returns:
        float: Peak strain amplitude (dimensionless)
    """
    return np.sqrt((4 * G_N * sr_rate**2) / (r**2 * omega_trans * trans_rate))

# --------------- superradiance growth rate calculation --------------

def calc_superradiance_rate(l, m, n, a_star, r_g, alpha):
    """
    Calculate superradiance growth rate Gamma for a given mode.
    
    Args:
        l (int): Orbital angular momentum quantum number (dimensionless)
        m (int): Azimuthal quantum number (dimensionless)
        n (int): Principal quantum number (dimensionless)
        a_star (float): Dimensionless spin parameter (dimensionless)
        r_g (float): Gravitational radius [eV^-1]
        mu_a (float): Axion mass [eV]
    
    Returns:
        float: Superradiance growth rate [eV] (inverse time in natural units)
    
    Notes:
        Returns the superradiance rate in natural units (ℏ = c = 1).
        For superradiance to occur: m * omega_plus > mu_a
        Based on non-relativistic approx from {https://arxiv.org/pdf/1004.3558}{Exploring the axiverse with precision black hole physics - Arvanitaki, Dubovsky}
    """
    import math
    
    # Internal helper functions
    def _calc_r_plus(r_g, a):
        """Calculate outer horizon radius."""
        return r_g + np.sqrt(r_g**2 - a**2)
    
    def _calc_omega_plus(r_g, a):
        """Calculate angular velocity at the outer horizon."""
        first = 1 / (2 * r_g)
        num = a / r_g
        denom = 1 + np.sqrt(1 - (a / r_g)**2)
        second = num / denom
        return first * second
    
    def _calc_clmn(l, m, n, a, r_g, alpha, r_plus, omega_plus):
        """Calculate the C_lmn coefficient."""
        num_1 = (2 ** (4 * l + 4)) * math.factorial(2 * l + n + 1)
        denom_1 = (l + n + 1) ** (2 * l + 4) * math.factorial(n)
        first = num_1 / denom_1
        
        num_2 = math.factorial(l)
        denom_2 = math.factorial(2 * l) * math.factorial(2 * l + 1)
        second = (num_2 / denom_2) ** 2
        
        prod = 1.0
        for j in range(1, l + 1):
            term = (j ** 2) * (1 - a ** 2 / r_g ** 2) + 4 * r_plus ** 2 * (m * omega_plus - alpha/r_g) ** 2
            prod *= term
        
        return first * second * prod
    
    # Main calculation
    a = a_star * r_g  # [eV^-1]
    r_plus = _calc_r_plus(r_g, a)  # [eV^-1]
    omega_plus = _calc_omega_plus(r_g, a)  # [eV]
    C_lmn = _calc_clmn(l, m, n, a, r_g, alpha, r_plus, omega_plus)  # dimensionless
    
    # Superradiance rate [eV]
    gamma = 2 * alpha/r_g * alpha ** (4 * l + 4) * r_plus * (m * omega_plus - alpha/r_g) * C_lmn
    
    return gamma

# --------------- characteristic time scale calculations --------------

def calc_char_t_ann(ann_rate, n_max):
    """
    Calculate characteristic timescale for annihilation.
    
    Args:
        ann_rate (float): Annihilation rate [eV]
        n_max (float): Maximum occupation number (dimensionless)
    
    Returns:
        float: Characteristic timescale [eV^-1] (time in natural units)
    """
    return 1/(n_max * ann_rate)

def calc_char_t_tran(sr_rate):
    """
    Calculate characteristic timescale for transition.
    
    Args:
        sr_rate (float): Superradiance rate [eV]
    
    Returns:
        float: Characteristic timescale [eV^-1] (time in natural units)
    """
    return 1/sr_rate

# --------------- various stuff needed to calculate event rate ---------------

def calc_astar_crit(r_g, alpha, n, m):
    """
    Calculate critical spin parameter for superradiance.
    
    Args:
        r_g (float): Gravitational radius [eV^-1]
        alpha (float): Fine structure constant (dimensionless)
        n (int): Principal quantum number (dimensionless)
        m (int): Azimuthal quantum number (dimensionless)
    
    Returns:
        float: Critical spin parameter (dimensionless)
    
    Notes:
        Superradiance occurs when a_star > a_star_crit.
        Formula uses axion frequency omega_axion = mu_a * (1 - alpha^2/(2*n^2))
    """
    omega_axion = alpha/r_g * (1-alpha**2/(2*n**2))  # [eV]
    astar_crit = 4 * r_g * omega_axion / (m * (1 + 4*r_g**2*omega_axion**2 / (m**2)))  # dimensionless
    return astar_crit

def calc_delta_astar(astar_init, r_g, alpha, n, m):
    """
    Calculate spin parameter difference (initial minus critical).
    
    Args:
        astar_init (float): Initial dimensionless spin parameter (dimensionless)
        r_g (float): Gravitational radius [eV^-1]
        alpha (float): Fine structure constant (dimensionless)
        n (int): Principal quantum number (dimensionless)
        m (int): Azimuthal quantum number (dimensionless)
    
    Returns:
        float: Spin parameter difference (dimensionless)
    
    Notes:
        Returns astar_init - astar_crit. Positive values indicate superradiance is possible.
    """
    astar_crit = calc_astar_crit(r_g, alpha, n, m)
    return astar_init - astar_crit

def calc_merger_rate(f_supress, f_pbh, r_g1, r_g2, f_m1, f_m2):
    """
    Calculate primordial black hole merger rate.
    
    Args:
        f_supress (float): Suppression factor (dimensionless)
        f_pbh (float): PBH dark matter fraction (dimensionless)
        r_g1 (float): Gravitational radius of first BH [eV^-1]
        r_g2 (float): Gravitational radius of second BH [eV^-1]
        f_m1 (float): Mass function value for first BH (dimensionless)
        f_m2 (float): Mass function value for second BH (dimensionless)
    
    Returns:
        float: Merger rate [Gpc^-3 yr^-1]
    
    Notes:
        Normalisation constant 7.481e9 corresponds to r_g for 1 M_☉ in [eV^-1].
        This formula was taken from {https://arxiv.org/pdf/2110.06188}{Miller, Aggarwal, Clesse, Lillo - Constraints on planetary and asteroid-mass primordial black holes from continuous gravitational-wave searches}.
    """
    symm_mass_ratio = r_g1 * r_g2 / (r_g1+r_g2)**2  # dimensionless
    normalised_mass = (r_g1 + r_g2)/(7.481e9)  # dimensionless, normalized to 1 M_☉
    return 1.6e6 * f_supress * f_pbh**(53/37) * normalised_mass**(-32/37) * symm_mass_ratio**(-34/37) * f_m1 * f_m2


def calc_event_rate_ann(r_g, delta_astar, m, h_det, G_N, ann_rate, omega, merger_rate):
    """
    Calculate event rate for annihilation-based gravitational wave detection.
    
    Args:
        r_g (float): Gravitational radius [eV^-1]
        delta_astar (float): Spin parameter difference (dimensionless)
        m (int): Azimuthal quantum number (dimensionless)
        h_det (float): Detector strain threshold (dimensionless)
        G_N (float): Gravitational constant [eV^-2]
        ann_rate (float): Annihilation rate [eV]
        omega (float): Annihilation frequency [eV]
        merger_rate (float): Merger rate [eV^4] (number per volume per time in natural units)
    
    Returns:
        float: Detection event rate [eV^-1] (inverse time in natural units)
    
    Notes:
        All units are in natural units (ℏ = c = 1).
        To convert to SI: multiply by ℏ [eV·s] to get events per second.
    """
    root = np.sqrt(8*ann_rate/(omega*G_N))  # [eV]
    parentheses = r_g**2 * delta_astar/(m*h_det) * root  # [eV^-1]
    return 4/3 * np.pi * parentheses**3 * merger_rate  # [eV^-1]

def calc_event_rate_tran(h_det, G_N, tran_rate, sr_rate, omega, merger_rate):
    """
    Calculate event rate for transition-based gravitational wave detection.
    
    Args:
        h_det (float): Detector strain threshold (dimensionless)
        G_N (float): Gravitational constant [eV^-2]
        tran_rate (float): Transition rate [eV]
        sr_rate (float): Superradiance rate [eV]
        omega (float): Transition frequency [eV]
        merger_rate (float): Merger rate [eV^4] (number per volume per time in natural units)
    
    Returns:
        float: Detection event rate [eV^-1] (inverse time in natural units)
    
    Notes:
        All units are in natural units (ℏ = c = 1).
        To convert to SI: multiply by ℏ [eV·s] to get events per second.
        Formula computes detection volume from h_peak = sqrt(4*G_N*sr_rate^2 / (r^2*omega*tran_rate))
    """
    parentheses = 4*G_N * sr_rate**2 / (h_det**2 * omega * tran_rate)  # [eV^-2] (r^2 in natural units)
    return 4/3 * np.pi * parentheses**(3/2) * merger_rate  # [eV^-1] (events per unit time)
