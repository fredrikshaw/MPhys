from cytools import fetch_polytopes
import matplotlib.pyplot as plt
import numpy as np
from scipy.differentiate import hessian
from scipy.linalg import eigh
from typing import Sequence
from tqdm import tqdm
import pickle
import os

h11=100
n_polys=10

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


def find_masses(polytope, silent=True, return_masses=True, return_potential=False, plot_hessian=False, plot_potential=False):
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

    # Now we need the charge matrix (whatever that is)
    q_list = []

    for div_index in prime_divisors:
        q_list.append(basis[:, div_index])
    
    q_list = np.array(q_list)
    if not silent:
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

    # Plot 1: Potential along diagonal (all thetas equal)
    if plot_potential:
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

    eigvals, eigvecs = eigh(H, g)  # This is solving the generalised eigenvalue problem which avoids the need to actively transform to canonical basis

    sqrt_lambda = np.where(eigvals >= 0, np.sqrt(eigvals), np.nan)
    masses_ev = sqrt_lambda / M_pl_in_ev
    if not silent:
        print(f"Mean mass [eV]: {np.mean(masses_ev)}")
        print(f"Mass range [eV]: {np.min(masses_ev)} - {np.max(np.masses_ev)}, min non-zero val: {np.min([m for m in masses_ev if m!=0])}")

        print("Rank(q):", np.linalg.matrix_rank(q_list))
        print("Min exp suppression:", np.min(np.exp(-2*np.pi * q_list @ tau)))
        print("Hessian norm:", np.linalg.norm(H))
        qa_dot_tau = q_list @ tau
        print("q·tau stats:", np.min(qa_dot_tau), np.max(qa_dot_tau))

    if plot_hessian:
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
    
    if return_masses and return_potential:
        return masses_ev, pot_wrapper
    elif return_masses:
        return masses_ev
    elif return_potential:
        return pot_wrapper
    else:
        return None

# Get all the polytopes we want, I think these are drawn randomly from KS database
polytopes = fetch_polytopes(h11=h11, lattice="N", favorable=True, limit=n_polys)

mass_list = []  # Just have a big ol' chunky list of all the masses to histogram
for polytope in tqdm(polytopes, desc="Processing polytopes"):
    mass_list.append(find_masses(polytope))

# Flatten the list of arrays and remove NaNs
all_masses = np.concatenate([m[~np.isnan(m)] for m in mass_list if m is not None])

# Save masses to pickle file (in same directory as this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
pickle_dir = os.path.join(script_dir, 'mass_dist_pickles')
os.makedirs(pickle_dir, exist_ok=True)
pickle_filename = os.path.join(pickle_dir, f'masses_h11_{h11}_npolys_{n_polys}_nmasses_{len(all_masses)}.pkl')
with open(pickle_filename, 'wb') as f:
    pickle.dump(all_masses, f)
print(f"Saved masses to {pickle_filename}")

# Plot histogram with logarithmic x-axis and normalized y-axis
plt.figure(figsize=(12, 7))
# Create logarithmically spaced bins
mass_min = np.min(all_masses[all_masses > 0])  # Exclude zeros for log scale
mass_max = np.max(all_masses)
log_bins = np.logspace(np.log10(mass_min), np.log10(mass_max), 51)  # 51 edges = 50 bins
plt.hist(all_masses, bins=log_bins, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Axion Mass [eV]', fontsize=12)
plt.ylabel('N', fontsize=12)
plt.title(f'Distribution of Axion Masses (N = {len(all_masses)} masses from {n_polys} polytopes with h11={h11})', 
          fontsize=14, fontweight='bold')
plt.xscale('log')
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.show()



