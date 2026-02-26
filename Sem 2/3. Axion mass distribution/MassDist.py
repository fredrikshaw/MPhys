from cytools import fetch_polytopes
import matplotlib.pyplot as plt
import numpy as np
from scipy.differentiate import hessian
from scipy.linalg import eigh
from typing import Sequence
from tqdm import tqdm

h11=100
n_polys=1

M_pl_in_ev = 1.22e28

def axion_potential(theta: Sequence[float],
                    q: np.ndarray,
                    tau: Sequence[float],
                    W0: float,
                    g_inv: np.ndarray,
                    V_vol: float) -> float:
    """
    Copying potential from https://arxiv.org/pdf/1808.01282
    ChatGPT wrote the function, but I have manually gone through and validated it
    Evaluate the potential
      V = -8*pi / V_vol^2 * [ sum_alpha q_alpha^i tau_i W0 e^{-2pi q_alpha.tau} cos(2pi q_alpha.theta)
                              + sum_{alpha>alpha'} ( pi * q_alpha^T g_inv q_alpha'
                                                   + (q_alpha + q_alpha') . tau )
                                * e^{-2pi (q_alpha+q_alpha').tau} * cos( 2pi theta . (q_alpha - q_alpha') )
                            ]
    Arguments:
      theta : array-like, shape (n_axions,)  -- axion angles (units where period = 1)
      q     : ndarray, shape (n_inst, n_axions) -- instanton charge matrix q_alpha^i
      tau   : array-like, shape (n_axions,)  -- saxions (4-cycle volumes)
      W0    : float -- flux superpotential constant
      g_inv  : ndarray, shape (n_axions, n_axions) -- inverse Kähler metric on axion sector
      V_vol : float -- overall Calabi-Yau volume mathcal{V}
    Returns:
      scalar float = V
    """
    theta = np.asarray(theta, dtype=float)
    q = np.asarray(q, dtype=float)               # shape (n_inst, n)
    tau = np.asarray(tau, dtype=float)           # shape (n,)
    g_inv = np.asarray(g_inv, dtype=float)         # shape (n,n)

    assert q.ndim == 2
    n_inst, n = q.shape
    assert theta.shape == (n,)
    assert tau.shape == (n,)
    assert g_inv.shape == (n, n)
    assert V_vol != 0

    # first sum: over instantons alpha
    qa_dot_tau = q.dot(tau)        # shape (n_inst,)
    qa_dot_theta = q.dot(theta)    # shape (n_inst,)

    term1 = np.sum((qa_dot_tau) * W0 * np.exp(-2*np.pi * qa_dot_tau) * np.cos(2*np.pi * qa_dot_theta))

    # second double-sum: alpha > alpha'
    term2 = 0.0
    for a in range(n_inst):
        qa = q[a]
        for b in range(a):
            qb = q[b]
            qsum = qa + qb                     # vector (n,)
            qdiff = qa - qb                    # vector (n,)
            exp_arg = -2*np.pi * qsum.dot(tau) # scalar
            cos_arg = 2*np.pi * qdiff.dot(theta)

            # bracket: pi * q_a^T g_inv q_b + (q_a + q_b) . tau
            bracket = np.pi * (qa @ g_inv @ qb) + qsum.dot(tau)

            term2 += bracket * np.exp(exp_arg) * np.cos(cos_arg)

    prefactor = -8.0 * np.pi / (V_vol**2)
    V = prefactor * (term1 + term2)
    # I'm gonna try the scaling that ChatGPT reckons the paper is using
    # SUGRA units where V \sim (M_pl)^4 * dimensionless function
    # This does make sense as they're in units where M_pl = 1 and it's a 4-d space
    V_phys = V * M_pl_in_ev**4
    return float(V_phys)

def numerical_hessian(f, x0, eps=1e-5):
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    H = np.zeros((n, n))

    # Exploit symmetry: only compute upper triangle (j >= i)
    for i in tqdm(range(n), desc="Computing Hessian"):
        for j in range(i, n):  # Only j >= i
            dx_i = np.zeros(n)
            dx_j = np.zeros(n)
            dx_i[i] = eps
            dx_j[j] = eps

            f_pp = f(x0 + dx_i + dx_j)
            f_pm = f(x0 + dx_i - dx_j)
            f_mp = f(x0 - dx_i + dx_j)
            f_mm = f(x0 - dx_i - dx_j)

            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
            
            # Copy to lower triangle (exploit symmetry)
            if i != j:
                H[j, i] = H[i, j]

    return H


# Get all the polytopes we want, I think these are drawn randomly from KS database
polytopes = fetch_polytopes(h11=h11, lattice="N", favorable=True, limit=n_polys)

for polytope in polytopes:
    # First find CY manifold for each polytope, we need to do this via triangulation
    t = polytope.triangulate(backend="topcom")
    cy = t.get_cy()

    # Now we want the intersection numbers \kappa_{ijk} to be able to compute the CY volume
    kappa = cy.intersection_numbers()

    # Now comes the point where we choose a random point in the cone
    # I can't remember if this is just cause of sampling and we would ideally integrate over all points
    # or if it's something else.
    # I think it might be to do with fixing the saxion value
    # It might implicitly set dV/dtau = 0
    cone = cy.toric_kahler_cone()
    point = cone.find_interior_point(check=True)

    # Compute the total colume of the CY space
    V = cy.compute_cy_volume(point)
    # Find the saxion field (is this literally just a number?)
    tau = cy.compute_divisor_volumes(point, in_basis=True)
    print(f"tau before fake normalisation: {tau}")
    # Seems to return list of 14 floats if in_basis=False, otherwise it returns list of 10
    # tau = tau/np.mean(tau)

    # Get the metric, might be handy later?
    g = cy.compute_kahler_metric(point)
    g_inv = cy.compute_inverse_kahler_metric(point)

    # Now we just need to compute the hessian and find the masses from that
    # Need to define a wrapper function for the potential
    # To do this we need to find q, the integer charge vector
    # Then should be able to calculate the Hessian straight from that
    # Then we should be able to get the masses as m^2 = g^-1 H, where g^-1 is to remove kinetic terms

    prime_divisors = cy.prime_toric_divisors() # We can use prime toric divisors as they're basically always rigid
    basis = cy.divisor_basis(as_matrix=True)

    print(f"This should be h11 (10 atm at least): {basis.shape[0]}")

    # Now we need the charge matrix (whatever that is)
    q_list = []

    for div_index in prime_divisors:
        q_list.append(basis[:, div_index])
    
    q_list = np.array(q_list)
    print("Charge matrix shape:", q_list.shape)
    print(q_list)
    print(f"\n tau: {tau}")

    def pot_wrapper(theta):
        theta = np.asarray(theta).reshape(-1)
        return axion_potential(theta=theta,
                               q=q_list,
                               tau=tau,
                               W0=1, # Approximation for classical flux superpotential used in 1808.0282
                               g_inv=g_inv,
                               V_vol=V)

    theta0 = np.zeros(h11)
    H = numerical_hessian(pot_wrapper, theta0)
    print(f"Printing Hessian:")
    for line in H:
        print(str(line).replace("\n", ""))
    print("\n")

    # Plot 1: Potential along diagonal (all thetas equal)
    theta_vals = np.linspace(-2*np.pi, 2*np.pi, 500)
    V_diagonal = [pot_wrapper(t * np.ones(h11)) for t in theta_vals]
    
    plt.figure(figsize=(10, 6))
    plt.plot(theta_vals, V_diagonal, 'b-', linewidth=2)
    plt.xlabel(r'$\theta$ (all $\theta_i$ equal) [radians]', fontsize=12)
    plt.ylabel(r'$V(\theta)$ [eV$^4$]', fontsize=12)
    plt.title(f'Axion Potential Along Diagonal (h$^{{1,1}}$ = {h11})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    eigvals, eigvecs = eigh(H, g)

    # print(f"Printing M^2:")
    # for line in M_squared:
    #     print(str(line).replace("\n", ""))
    # print("\n")

    # eigvals = np.linalg.eigvalsh(M_squared)
    masses_planck = np.sqrt(np.abs(eigvals))
    masses_ev = masses_planck / M_pl_in_ev
    print(f"Masses [M_pl]: {masses_planck}")
    print(f"Masses [eV]: {masses_ev}")

    print("Rank(q):", np.linalg.matrix_rank(q_list))
    print("Min exp suppression:", np.min(np.exp(-2*np.pi * q_list @ tau)))
    print("Hessian norm:", np.linalg.norm(H))
    qa_dot_tau = q_list @ tau
    print("q·tau stats:", np.min(qa_dot_tau), np.max(qa_dot_tau))

    # Plot the Hessian matrix
    plt.figure(figsize=(10, 8))
    im = plt.imshow(H, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    plt.colorbar(im, label=r'Hessian Element Value [eV$^4$]')
    plt.xlabel('Axion Index $j$', fontsize=12)
    plt.ylabel('Axion Index $i$', fontsize=12)
    plt.title(r'Hessian Matrix $H_{ij} = \frac{\partial^2 V}{\partial \theta_i \partial \theta_j}$', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

