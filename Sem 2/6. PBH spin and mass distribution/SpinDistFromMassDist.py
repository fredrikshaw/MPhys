"""
Matched final-spin models as a function of symmetric mass ratio nu.

This script compares:
1) Large-nu fit (valid near nu ~ 0.25):  a_f = 0.13 + 2.24*nu
2) Small-nu expansion (valid for nu << 1): a_f = 2*sqrt(3)*nu - 3.87*nu^2
3) A smooth matched composite between the two limits
4) The self-consistent spin from SimplePostMergerSpin.py
"""

import numpy as np
import matplotlib.pyplot as plt

from SimplePostMergerSpin import solve_a_star_self_consistent


def a_star_high_nu(nu):
	"""Final spin fit valid in the nu ~ 0.25 limit (arXiv:1903.01179)."""
	return 0.13 + 2.24 * nu


def a_star_low_nu(nu):
	"""Final spin expansion valid in the nu << 1 limit (arXiv:1605.01938)."""
	return 2 * np.sqrt(3) * nu - 3.87 * nu**2


def smoothstep(x):
	"""C1 smooth transition function with values clipped to [0, 1]."""
	x_clipped = np.clip(x, 0.0, 1.0)
	return x_clipped**2 * (3 - 2 * x_clipped)


def matched_a_star(nu, nu_low=0.10, nu_high=0.20):
	"""
	Smoothly match low-nu and high-nu approximations.

	For nu <= nu_low   : use low-nu expression
	For nu >= nu_high  : use high-nu expression
	Between the two    : smooth interpolation
	"""
	nu = np.asarray(nu)

	if nu_low >= nu_high:
		raise ValueError("nu_low must be smaller than nu_high")

	low_val = a_star_low_nu(nu)
	high_val = a_star_high_nu(nu)

	t = (nu - nu_low) / (nu_high - nu_low)
	w = smoothstep(t)

	matched = (1 - w) * low_val + w * high_val
	matched = np.where(nu <= nu_low, low_val, matched)
	matched = np.where(nu >= nu_high, high_val, matched)

	return matched


def sample_lognormal_masses(M_central, sigma, N):
	"""Sample N masses from a lognormal distribution."""
	return np.random.lognormal(mean=np.log(M_central), sigma=sigma, size=N)


def sample_merger_nu_lognormal(N_mergers, M_central, sigma):
	"""
	Sample N_mergers by drawing 2N masses and pairing consecutive points.

	Returns
	-------
	tuple[np.ndarray, np.ndarray, np.ndarray]
		(nu_values, m1_values, m2_values)
	"""
	if N_mergers <= 0:
		raise ValueError("N_mergers must be positive")

	masses = sample_lognormal_masses(M_central=M_central, sigma=sigma, N=2 * N_mergers)
	m1_values = masses[0::2]
	m2_values = masses[1::2]
	M_tot = m1_values + m2_values
	nu_values = (m1_values * m2_values) / (M_tot**2)

	return nu_values, m1_values, m2_values


def plot_matched_spin_histogram_from_lognormal_mergers(
	N_mergers=10000,
	M_central=30.0,
	sigma_values=None,
	bins=60,
	nu_low_valid_max=0.10,
	nu_high_valid_min=0.20,
	include_mass_hist=False,
	mass_xlim_quantile=0.999,
	log_y=False,
	log_floor=1e-3,
	save_fig=False,
	filename="matched_spin_hist_lognormal.png",
	show=True,
):
	"""
	Sample merger masses from a lognormal distribution and histogram matched spins.

	Plots overlaid histograms (outline only) for multiple sigma values.

	Parameters
	----------
	N_mergers : int, optional
		Number of mergers to sample. Default 10000
	M_central : float, optional
		Central mass for lognormal distribution. Default 30.0
	sigma_values : list of float, optional
		List of sigma values to plot. Default is [0.1, 0.5, 1.0]
	bins : int, optional
		Number of bins for histograms. Default 60
	nu_low_valid_max : float, optional
		Upper bound of low-nu validity region. Default 0.10
	nu_high_valid_min : float, optional
		Lower bound of high-nu validity region. Default 0.20
	include_mass_hist : bool, optional
		Whether to include mass histogram. Default False
	mass_xlim_quantile : float or None, optional
		If set, clip the mass-plot x-axis at this quantile of all sampled masses
		to suppress far-right outlier tails (e.g. 0.995 keeps 99.5% of samples).
		Set to None to disable. Default 0.995
	log_y : bool, optional
		If True, use logarithmic y-axis. Default False
	log_floor : float, optional
		Minimum y value used when log_y=True to avoid log(0). Default 1e-3
	save_fig : bool, optional
		Whether to save the figure. Default False
	filename : str, optional
		Filename for saved figure. Default "matched_spin_hist_lognormal.png"
	show : bool, optional
		Whether to display the figure. Default True
	"""
	if sigma_values is None:
		sigma_values = [0.1, 0.5, 1.0]

	default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
	colors = [default_cycle[i % len(default_cycle)] for i in range(len(sigma_values))]
	n_sci = f"{N_mergers:.2e}"
	two_n_sci = f"{2 * N_mergers:.2e}"
	if log_floor <= 0:
		raise ValueError("log_floor must be positive")
	if mass_xlim_quantile is not None and not (0 < mass_xlim_quantile < 1):
		raise ValueError("mass_xlim_quantile must be in (0, 1) or None")

	if include_mass_hist:
		fig, axes = plt.subplots(2, 1, figsize=(10, 10))
		ax_spin, ax_mass = axes
	else:
		fig, ax_spin = plt.subplots(figsize=(10, 6))

	# Compute all histograms first for independent normalization
	spin_hists = []
	mass_hists = []
	mass_samples_all = []

	for idx, sigma in enumerate(sigma_values):
		nu_values, m1_values, m2_values = sample_merger_nu_lognormal(
			N_mergers=N_mergers,
			M_central=M_central,
			sigma=sigma,
		)

		a_star_values = matched_a_star(
			nu_values,
			nu_low=nu_low_valid_max,
			nu_high=nu_high_valid_min,
		)

		# Get histogram data and normalize to own maximum
		counts_spin, edges_spin = np.histogram(a_star_values, bins=bins, density=True)
		normalized_spin = counts_spin / np.max(counts_spin)
		spin_hists.append((normalized_spin, edges_spin))

		if include_mass_hist:
			all_masses = np.concatenate([m1_values, m2_values])
			mass_samples_all.append(all_masses)
			counts_mass, edges_mass = np.histogram(all_masses, bins=bins, density=True)
			normalized_mass = counts_mass / np.max(counts_mass)
			mass_hists.append((normalized_mass, edges_mass))

	# Plot normalized spin histograms
	for idx, sigma in enumerate(sigma_values):
		normalized_counts, edges_spin = spin_hists[idx]
		if log_y:
			normalized_counts = np.clip(normalized_counts, log_floor, None)
		ax_spin.stairs(
			normalized_counts,
			edges_spin,
			color=colors[idx],
			linewidth=1.5,
			label=rf"$\sigma={sigma}$",
		)

	ax_spin.set_xlabel(r"Matched final spin $a_{*,f}$", fontsize=12)
	ax_spin.set_ylabel("Peak-normalized value", fontsize=12)
	ax_spin.set_title(
		rf"Peak-normalized histogram of matched $a_{{*,f}}$ "
		rf"($N={n_sci}$, $M_c={M_central}$)",
		fontsize=13,
	)
	if log_y:
		ax_spin.set_yscale("log")
		ax_spin.set_ylim(log_floor, 1.1)
	ax_spin.grid(alpha=0.3)
	ax_spin.legend(fontsize=10)

	# Plot normalized mass histograms
	if include_mass_hist:
		if mass_xlim_quantile is not None and len(mass_samples_all) > 0:
			all_mass_values = np.concatenate(mass_samples_all)
			xmax = np.quantile(all_mass_values, mass_xlim_quantile)
			ax_mass.set_xlim(left=0, right=xmax)

		for idx, sigma in enumerate(sigma_values):
			normalized_counts, edges_mass = mass_hists[idx]
			if log_y:
				normalized_counts = np.clip(normalized_counts, log_floor, None)
			ax_mass.stairs(
				normalized_counts,
				edges_mass,
				color=colors[idx],
				linewidth=1.5,
				label=rf"$\sigma={sigma}$",
			)

		ax_mass.set_xlabel("Sampled mass", fontsize=12)
		ax_mass.set_ylabel("Peak-normalized value", fontsize=12)
		ax_mass.set_title(
			rf"Peak-normalized histogram of sampled mass "
			rf"($2N={two_n_sci}$ samples, $M_c={M_central}$)",
			fontsize=13,
		)
		if log_y:
			ax_mass.set_yscale("log")
			ax_mass.set_ylim(log_floor, 1.1)
		ax_mass.grid(alpha=0.3)
		ax_mass.legend(fontsize=10)

	plt.tight_layout()

	if save_fig:
		plt.savefig(filename, dpi=300, bbox_inches="tight")
		print(f"Saved figure: {filename}")

	if show:
		plt.show()
	else:
		plt.close(fig)

	return a_star_values, nu_values


def bkl_curve(nu_values):
	"""Compute the self-consistent spin curve from SimplePostMergerSpin.py."""
	spins = []
	converged_flags = []

	for nu in nu_values:
		result = solve_a_star_self_consistent(nu, prograde=True)
		spins.append(result["a_star"])
		converged_flags.append(result["converged"])

	return np.array(spins), np.array(converged_flags, dtype=bool)


def plot_all_a_star_models(
	nu_min=0.005,
	nu_max=0.25,
	num_points=500,
	nu_low_valid_max=0.10,
	nu_high_valid_min=0.20,
	save_fig=False,
	filename="a_star_f_vs_nu_matched.png",
	show=True,
):
	"""Plot asymptotic, matched, and BKL final spin models on one figure."""
	nu_values = np.linspace(nu_min, nu_max, num_points)

	low_curve = a_star_low_nu(nu_values)
	high_curve = a_star_high_nu(nu_values)
	matched_curve = matched_a_star(nu_values, nu_low=nu_low_valid_max, nu_high=nu_high_valid_min)
	bkl_spins, bkl_converged = bkl_curve(nu_values)

	fig, ax = plt.subplots(figsize=(10, 6))

	ax.plot(nu_values, low_curve, "--", color="tab:blue", linewidth=2,
			label=r"Small-$\nu$ expansion: $2\sqrt{3}\,\nu - 3.87\,\nu^2$")
	ax.plot(nu_values, high_curve, "--", color="tab:orange", linewidth=2,
			label=r"Large-$\nu$ fit: $0.13 + 2.24\,\nu$")
	ax.plot(nu_values, matched_curve, "-", color="tab:green", linewidth=2.5,
			label="Matched asymptotic model")
	ax.plot(nu_values, bkl_spins, "-", color="tab:red", linewidth=2,
			label="Self-consistent ISCO model (SimplePostMergerSpin)")

	if not np.all(bkl_converged):
		failed = nu_values[~bkl_converged]
		ax.scatter(
			failed,
			bkl_spins[~bkl_converged],
			color="black",
			marker="x",
			s=25,
			label="BKL non-converged points",
		)

	ax.axvspan(nu_min, nu_low_valid_max, alpha=0.12, color="tab:blue", label="Small-$\\nu$ validity")
	ax.axvspan(nu_low_valid_max, nu_high_valid_min, alpha=0.10, color="tab:green", label="Matching region")
	ax.axvspan(nu_high_valid_min, nu_max, alpha=0.12, color="tab:orange", label="Large-$\\nu$ validity")

	ax.set_xlim(nu_min, nu_max)
	ax.set_xlabel(r"Symmetric mass ratio $\nu = m_1 m_2 / (m_1+m_2)^2$", fontsize=12)
	ax.set_ylabel(r"Final spin $a_{*,f}$", fontsize=12)
	ax.set_title(r"Final Spin $a_{*,f}$ vs $\nu$: Asymptotics, Matching, and ISCO Model", fontsize=13)
	ax.grid(alpha=0.3)
	ax.legend(fontsize=9, loc="best")

	plt.tight_layout()

	if save_fig:
		plt.savefig(filename, dpi=300, bbox_inches="tight")
		print(f"Saved figure: {filename}")

	if show:
		plt.show()
	else:
		plt.close(fig)


if __name__ == "__main__":
	plot_all_a_star_models(save_fig=False, show=False)
	plot_matched_spin_histogram_from_lognormal_mergers(
		N_mergers=2000000,
		M_central=1e6,
		sigma_values=[0.1, 0.5, 1.0],
		bins=500,
		save_fig=False,
		include_mass_hist=True,
		log_y=True,
		show=True,
	)
