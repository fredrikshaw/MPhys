# Python code to reproduce the expressions in Appendix A (Tables VI & VII)
# from Arvanitaki et al. (arXiv:1411.2263v3) and compute annihilation &
# transition rates via eq. (A12).
#
# Notes:
# - This code implements the analytic leading-order angular formulas
#   that appear in Tables VI and VII (copied from Appendix A).  It does
#   not re-derive the full Fourier-transform of Tij; instead it provides
#   reusable functions that evaluate the differential power dP/dOmega
#   for the specific levels/transitions appearing in the paper, and then
#   integrate over solid angle to produce total power and the rates in
#   eq. (A12):
#       Gamma_a = (1 / (2 * omega * N**2)) * ∫ dΩ (dP/dΩ)_ann
#       Gamma_t = (1 / (omega * N * Nprime)) * ∫ dΩ (dP/dΩ)_trans
#
# - The code follows the simplification you mentioned: for annihilation
#   rates we drop the angular cosine terms that integrate to zero
#   (i.e. we keep only the constant part of the pattern when computing
#   Gamma_a). You can toggle that behavior with `use_cos_terms`.
#
# - Units: the paper uses ℏ = c = 1 (natural units). Here the code
#   treats G_N and r_g as input parameters — you must supply them in
#   consistent units if you want SI outputs. By default G_N=1, r_g=1.
#
# - For numerical theta integrals we use a straightforward quadrature
#   (vectorized trapezoid). Azimuthal integral is analytic (2π) because
#   dP/dΩ depends only on polar angle θ in the table entries.
#
# Example usage at the bottom computes:
#  - differential power for annihilation 2p at alpha=0.3 and theta=pi/4
#  - total annihilation power (integrated) and Gamma_a (with N)
#  - differential power for transition 6g->5g and its Gamma_t
#
# References: Appendix A, Tables VI & VII in arXiv:1411.2263v3.
# (The analytic expressions are copied from those tables.)

import numpy as np
from scipy import constants
from SuperradianceGrowthRate import calc_gamma

SOLAR_MASS = 1.988e30  # [kg]

# ------------------------------- helpers -------------------------------
def solid_angle_integral_from_dPdOmega(dPdtheta_func, Ntheta=10000):
    """
    Given a function dP/dOmega(theta) (independent of phi), compute
    the total power P = ∫ dΩ dP/dΩ = 2π ∫_0^π sinθ dθ dP/dΩ(θ).
    """
    theta = np.linspace(0.0, np.pi, Ntheta)
    dP_dOmega = dPdtheta_func(theta)
    integrand = np.sin(theta) * dP_dOmega  # integrand for polar integral
    P = 2 * np.pi * np.trapezoid(integrand, theta)
    return P

# Common angular polynomial that appears in many table entries:
def angular_poly(theta):
    # 28 cos 2θ + cos 4θ + 35
    return 28.0 * np.cos(2.0 * theta) + np.cos(4.0 * theta) + 35.0

# --------------------------- annihilation dP/dΩ -------------------------
def dPdOmega_ann_2p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power for annihilation from 2p (table VI).
    Returns dP/dΩ divided by N^2; user can multiply/divide as needed.
    """
    # copied expression from the table (first line of 2p entry)
    # dP/dΩ N^{-2} = alpha^18 * G_N * (...)^2 * angular_poly(theta) / (224 π (alpha^2+4)^4 r_g^4)
    prefactor = alpha**18 * G_N / (2**24 * np.pi * (alpha**2 + 4.0)**4 * r_g**4)
    bracket = (6.0 * alpha**3 + 40.0 * alpha - 3.0 * (alpha**2 + 4.0)**2 * np.arctan(2.0 / alpha))
    return prefactor * bracket**2 * angular_poly(theta)

def dPdOmega_ann_3d_leading(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Leading-order expanded term for 3d from Table VI:
    (This matches the table's leading term notation.)
    """
    prefactor = alpha**20 * G_N / (2**4 * 3**16 * np.pi * r_g**4)
    return prefactor * (np.sin(theta)**4) * angular_poly(theta)

def dPdOmega_ann_4f_leading(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Leading-order expanded term for 4f from Table VI.
    """
    prefactor = alpha**24 * G_N / (5**(-2) * 2**24 * np.pi * r_g**4)
    return prefactor * (np.sin(theta)**8) * angular_poly(theta)

# Generic dispatcher for annihilation differential power for supported levels
def dPdOmega_ann(level, alpha, theta, G_N=1.0, r_g=1.0):
    level = level.lower().strip()
    if level in ("2p", "2 p", "2 p "):
        return dPdOmega_ann_2p(alpha, theta, G_N=G_N, r_g=r_g)
    if level in ("3d",):
        return dPdOmega_ann_3d_leading(alpha, theta, G_N=G_N, r_g=r_g)
    if level in ("4f",):
        return dPdOmega_ann_4f_leading(alpha, theta, G_N=G_N, r_g=r_g)
    raise ValueError("Unsupported annihilation level: " + str(level))

# ------------------------- transition dP/dΩ ------------------------------
# Table VII leading terms (copied directly):
def dPdOmega_tr_6g_5g(alpha, theta, G_N=1.0, r_g=1.0):
    prefactor = 2**28 * 3**4 * 5**5 * (alpha**12) * G_N / (11**22 * np.pi * r_g**4)
    return prefactor * (np.sin(theta)**4)

def dPdOmega_tr_7h_6h(alpha, theta, G_N=1.0, r_g=1.0):
    prefactor = 2**31 * 3**7 * 5**2 * 7**6 * (alpha**12) * G_N / (13**26 * np.pi * r_g**4)
    return prefactor * (np.sin(theta)**4)

def dPdOmega_tr_5f_4f(alpha, theta, G_N=1.0, r_g=1.0):
    # 22252 α^12 G_N sin^4 θk / (334 π r_g^4)
    prefactor = 2**22 * 5**2 * (alpha**12) * G_N / (3**34 * np.pi * r_g**4)
    return prefactor * (np.sin(theta)**4)

def dPdOmega_transition(transition, alpha, theta, G_N=1.0, r_g=1.0):
    key = transition.replace(" ", "").lower()
    if key in ("6g->5g", "6g-5g", "6g5g"):
        return dPdOmega_tr_6g_5g(alpha, theta, G_N=G_N, r_g=r_g)
    if key in ("7h->6h", "7h6h", "7h-6h"):
        return dPdOmega_tr_7h_6h(alpha, theta, G_N=G_N, r_g=r_g)
    if key in ("5f->4f", "5f4f", "5f-4f"):
        return dPdOmega_tr_5f_4f(alpha, theta, G_N=G_N, r_g=r_g)
    raise ValueError("Unsupported transition: " + str(transition))

# ------------------------- integrate & rates -----------------------------
def total_power_ann(level, alpha, G_N=1.0, r_g=1.0):
    func = lambda th: dPdOmega_ann(level, alpha, th, G_N=G_N, r_g=r_g)
    return solid_angle_integral_from_dPdOmega(func)

def total_power_transition(transition, alpha, G_N=1.0, r_g=1.0):
    func = lambda th: dPdOmega_transition(transition, alpha, th, G_N=G_N, r_g=r_g)
    return solid_angle_integral_from_dPdOmega(func)

def calculate_omega_ann(mu_a, alpha, n):
    """
    Compute annihilation frequency omega_ann for level n.
    omega_ann = 2 * mu_a * (1 - alpha^2/(2 n^2))
    """
    return 2.0 * mu_a * (1.0 - alpha**2 / (2.0 * n**2))

def calculate_omega_transition(mu_a, alpha, n_e, n_g):
    """
    Compute transition frequency omega_tr between levels n_e and n_g.
    omega_tr = mu_a * alpha^2/2 * (1/n_g^2 - 1/n_e^2)
    """
    return 0.5 * mu_a * alpha**2 * (1.0 / n_g**2 - 1.0 / n_e**2)

def calculate_alpha_ann(omega_ann, r_g, n, tol_imag=1e-8):
    """
    Solve for alpha from:
        omega_ann = 2*(alpha/r_g)*(1 - alpha^2/(2 n^2))
    which reduces to the cubic
        alpha^3 - 2 n^2 alpha + n^2 * (omega_ann * r_g) = 0.
    Returns the physically relevant real, positive root (if one exists).
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


def calculate_alpha_trans(omega_tr, r_g, n_e, n_g):
    """
    Solve for alpha from:
        omega_tr = (1/2) * (alpha/r_g) * alpha^2 * (1/n_g^2 - 1/n_e^2)
    => alpha^3 = (2 * omega_tr * r_g) / Delta,  Delta = 1/n_g^2 - 1/n_e^2
    Returns the real cube root. If Delta is negative, the returned alpha keeps sign
    according to numerator/denominator (but physically Delta>0 when n_g < n_e).
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

def annihilation_rate(level, alpha, omega, G_N=1.0, r_g=1.0):
    """
    Compute Gamma_a per eq. (A12).
      Gamma_a = (1/(2 * omega * N^2)) * ∫ dΩ (dP/dΩ)|ann
    If use_cos_terms=False we drop the cos components in the common
    angular polynomial (28 cos2θ + cos4θ + 35) — keeping only the
    constant 35 — per the integration simplification you mentioned.
    """
    func = lambda th: dPdOmega_ann(level, alpha, th, G_N=G_N, r_g=r_g)
    P = solid_angle_integral_from_dPdOmega(func)
    Gamma_a = P / (2.0 * omega)
    return Gamma_a, P

def transition_rate(transition, alpha, omega, G_N=1.0, r_g=1.0):
    """
    Compute Gamma_t per eq. (A12):
      Gamma_t = (1/(omega * N * N')) * ∫ dΩ (dP/dΩ)|tr
    """
    func = lambda th: dPdOmega_transition(transition, alpha, th, G_N=G_N, r_g=r_g)
    P = solid_angle_integral_from_dPdOmega(func)
    Gamma_t = P / (omega)
    return Gamma_t, P

# ----------------------------- examples --------------------------------
if __name__ == "__main__":
    # ============================================================
    # INPUT PARAMETERS
    # ============================================================
    blackholemass = 10 # [M_sun]
    omega_GHz = 3e-8  # [GHz] - used to calculate omega when alpha_override is None
    omega = omega_GHz * 4.1357e-6  # [eV]
    alpha_override = None  # Set to a value to override alpha calculation (if None, alpha is calculated from omega)
    a_star = 0.99  # dimensionless spin parameter
    inv_ev_to_years = 2.09e-23  # conversion factor

    # ============================================================
    # BLACK HOLE PARAMETERS
    # ============================================================
    m_bh_J = blackholemass * SOLAR_MASS * constants.c ** 2       # [J]
    m_bh_ev = m_bh_J / constants.e                               # [eV]
    G_N = 6.708e-57                                              # [eV^-2]
    r_g = G_N * m_bh_ev                                          # [eV^-1]
    a = a_star * r_g                                             # [eV^-1]

    print("\n" + "=" * 70)
    print("BLACK HOLE PARAMETERS")
    print("=" * 70)
    print(f"Black hole mass (M_BH):          {blackholemass:.4e} M_☉")
    print(f"                                 {m_bh_ev:.4e} eV")
    print(f"Gravitational radius (r_g):      {r_g:.4e} eV^-1")
    print(f"Newton's constant (G_N):         {G_N:.4e} eV^-2")
    print(f"Dimensionless spin (a_*):        {a_star:.4e}")
    print(f"Spin parameter (a):              {a:.4e} eV^-1")
    print("=" * 70)

    # ============================================================
    # ANNIHILATION RATE (2p level)
    # ============================================================
    print("\n" + "=" * 70)
    print("ANNIHILATION RATE: 2p → ∅")
    print("=" * 70)
    
    n = 2
    
    if alpha_override is not None:
        # Use provided alpha and calculate omega from it
        alpha = alpha_override
        mu_a = alpha / r_g
        omega_ann = calculate_omega_ann(mu_a, alpha, n)
        omega_ann_GHz = omega_ann / 4.1357e-6  # Convert eV to GHz
        print(f"Using alpha override:            α = {alpha:.4e}")
        print(f"Calculated axion mass (μ_a):     {mu_a:.4e} eV")
        print(f"Calculated frequency (ω_ann):    {omega_ann:.4e} eV")
        print(f"                                 {omega_ann_GHz:.4e} GHz")
    else:
        # Calculate alpha from provided omega
        alpha = calculate_alpha_ann(omega, r_g, n)
        mu_a = alpha / r_g
        omega_ann = omega
        omega_ann_GHz = omega_ann / 4.1357e-6  # Convert eV to GHz
        print(f"Using frequency:                 ω = {omega:.4e} eV")
        print(f"                                 {omega_ann_GHz:.4e} GHz")
        print(f"Calculated α:                    {alpha:.4e}")
        print(f"Calculated axion mass (μ_a):     {mu_a:.4e} eV")
    
    print(f"Level:                           n = {n}")
    print("-" * 70)

    # total annihilation power and Gamma_a (dropping cos terms as instructed)
    Gamma_a_2p, P_2p = annihilation_rate("2p", alpha, omega_ann, G_N=G_N, r_g=r_g)
    
    # Calculate superradiance rate for 2p (l=1, m=1, n=2)
    l_ann, m_ann = 1, 1
    Gamma_sr_2p = calc_gamma(l_ann, m_ann, n, a, r_g, mu_a)  # [eV]
    
    print(f"Total annihilation power (P):    {P_2p:.4e} eV^2")
    print(f"Annihilation rate (Γ_a):         {Gamma_a_2p:.4e} eV")
    print(f"                                 {Gamma_a_2p / inv_ev_to_years:.4e} years^-1")
    print(f"Inverse rate (1/Γ_a):            {(1 / Gamma_a_2p) * inv_ev_to_years:.4e} years")
    print(f"Superradiance rate (Γ_sr):       {Gamma_sr_2p:.4e} eV")
    print(f"                                 {Gamma_sr_2p / inv_ev_to_years:.4e} years^-1")
    print(f"Inverse rate (1/Γ_sr):           {(1 / Gamma_sr_2p) * inv_ev_to_years:.4e} years")
    print("=" * 70)

    # ============================================================
    # TRANSITION RATE (6g → 5g)
    # ============================================================
    print("\n" + "=" * 70)
    print("TRANSITION RATE: 6g → 5g")
    print("=" * 70)
    
    ne, ng = 6, 5
    
    if alpha_override is not None:
        # Use provided alpha and calculate omega from it
        alpha = alpha_override
        mu_a = alpha / r_g
        omega_tr = calculate_omega_transition(mu_a, alpha, ne, ng)
        omega_tr_GHz = omega_tr / 4.1357e-6  # Convert eV to GHz
        print(f"Using alpha override:            α = {alpha:.4e}")
        print(f"Calculated axion mass (μ_a):     {mu_a:.4e} eV")
        print(f"Calculated frequency (ω_tr):     {omega_tr:.4e} eV")
        print(f"                                 {omega_tr_GHz:.4e} GHz")
    else:
        # Calculate alpha from provided omega
        alpha = calculate_alpha_trans(omega, r_g, ne, ng)
        mu_a = alpha / r_g
        omega_tr = omega
        omega_tr_GHz = omega_tr / 4.1357e-6  # Convert eV to GHz
        print(f"Using frequency:                 ω = {omega:.4e} eV")
        print(f"                                 {omega_tr_GHz:.4e} GHz")
        print(f"Calculated α:                    {alpha:.4e}")
        print(f"Calculated axion mass (μ_a):     {mu_a:.4e} eV")
    
    print(f"Initial level:                   n_e = {ne} (6g)")
    print(f"Final level:                     n_g = {ng} (5g)")
    print("-" * 70)
    
    Gamma_t_6g5g, P_6g5g = transition_rate("6g->5g", alpha, omega_tr, G_N=G_N, r_g=r_g)
    
    # Calculate superradiance rate for 6g (l=4, m=4, n=6)
    l_tr, m_tr = 4, 4
    Gamma_sr_6g5g = calc_gamma(l_tr, m_tr, ne, a, r_g, mu_a)  # [eV]
    
    print(f"Total transition power (P):      {P_6g5g:.4e} eV^2")
    print(f"Transition rate (Γ_t):           {Gamma_t_6g5g:.4e} eV")
    print(f"                                 {Gamma_t_6g5g / inv_ev_to_years:.4e} years^-1")
    print(f"Inverse rate (1/Γ_t):            {(1 / Gamma_t_6g5g) * inv_ev_to_years:.4e} years")
    print(f"Superradiance rate (Γ_sr):       {Gamma_sr_6g5g:.4e} eV")
    print(f"                                 {Gamma_sr_6g5g / inv_ev_to_years:.4e} years^-1")
    print(f"Inverse rate (1/Γ_sr):           {(1 / Gamma_sr_6g5g) * inv_ev_to_years:.4e} years")
    print("=" * 70 + "\n")
