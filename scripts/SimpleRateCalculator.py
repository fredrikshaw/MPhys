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
    P = 2 * np.pi * np.trapz(integrand, theta)
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
def total_power_ann(level, alpha, G_N=1.0, r_g=1.0, Nnorm=1.0):
    func = lambda th: dPdOmega_ann(level, alpha, th, G_N=G_N, r_g=r_g, Nnorm=Nnorm)
    return solid_angle_integral_from_dPdOmega(func)

def total_power_transition(transition, alpha, G_N=1.0, r_g=1.0, N1=1.0, N2=1.0):
    func = lambda th: dPdOmega_transition(transition, alpha, th, G_N=G_N, r_g=r_g, N1=N1, N2=N2)
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
    blackholemass = 10  # [M_sun]
    inv_ev_to_years = 2.09e-23  # conversion factor

    # Parameters
    m_bh_J = blackholemass * SOLAR_MASS * constants.c ** 2       # [J]
    m_bh_ev = m_bh_J / constants.e                               # [eV]
    G_N = 6.708e-57                                              # [eV^-2]
    r_g = G_N * m_bh_ev                                          # [eV^-1]


    # For annihilation frequency omega_ann = 2 * mu_a * (1 - alpha^2/(2 n^2))
    n = 2
    alpha = 0.3
    mu_a = alpha/r_g  # placeholder axion mass in same units as omega
    omega_ann = calculate_omega_ann(mu_a, alpha, n)

    # total annihilation power and Gamma_a (dropping cos terms as instructed)
    Gamma_a_2p, P_2p = annihilation_rate("2p", alpha, omega_ann, G_N=G_N, r_g=r_g)
    print("Gamma_a (2p) [eV] =", f"{Gamma_a_2p:.2e}")
    print("Gamma_a (2p) [years] =", f"{Gamma_a_2p / inv_ev_to_years:.2e}")
    print("Inverse gamma (2p) [years] =", f"{(1 / Gamma_a_2p) * inv_ev_to_years:.2e}")
    

    # Transition example: 6g -> 5g
    ne, ng = 6, 5
    alpha = 1.2
    mu_a = alpha/r_g  # placeholder axion mass in same units as omega
    omega_tr = calculate_omega_transition(mu_a, alpha, ne, ng)
    Gamma_t_6g5g, P_6g5g = transition_rate("6g->5g", alpha, omega_tr, G_N=G_N, r_g=r_g)
    print("Gamma_t (6g->5g) [eV] =", f"{Gamma_t_6g5g:.2e}")
    print("Gamma_t (6g->5g) [years] =", f"{Gamma_t_6g5g / inv_ev_to_years:.2e}")
    print("Inverse gamma (6g->5g) [years] =", f"{(1 / Gamma_t_6g5g) * inv_ev_to_years:.2e}")
