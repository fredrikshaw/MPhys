"""
PBH binary merger-rate utilities with optional spin-cutoff selection.

Implements the semi-analytic non-Poisson merger-rate model used in
Hutsi, Raidal, Vaskonen, Veermae (arXiv:2012.02786), with a symmetric-mass-ratio
cutoff:\n
    dR/(dm1 dm2) \\propto Theta(nu - nu_th) * nu^{-34/37} * S1 * S2 * psi(m1)psi(m2)

Masses are assumed to be in solar-mass units throughout.
"""

from __future__ import annotations

import math
import sys
from typing import Literal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

current_dir = Path(__file__).resolve().parent
sem2_dir = current_dir.parent
scripts_dir = sem2_dir / "0. Scripts from Sem 1"
if scripts_dir.exists():
    sys.path.insert(0, str(scripts_dir.resolve()))

from ParamCalculator import _trapezoid

try:
    from scipy.special import gamma as scipy_gamma
    from scipy.special import hyperu as scipy_hyperu
    _SCIPY_SPECIAL_AVAILABLE = True
except Exception:  # pragma: no cover
    _SCIPY_SPECIAL_AVAILABLE = False

try:
    from SpinDistFromMassDist import matched_a_star
    _MATCHED_SPIN_AVAILABLE = True
except Exception:  # pragma: no cover
    _MATCHED_SPIN_AVAILABLE = False

from SimplePostMergerSpin import solve_a_star_self_consistent


RATE_PREFactor_GPC3_YR = 1.6e6
DEFAULT_SIGMA_M = 0.004


def symmetric_mass_ratio(m1: np.ndarray | float, m2: np.ndarray | float) -> np.ndarray:
    """Return symmetric mass ratio nu = m1*m2/(m1+m2)^2."""
    m1_arr = np.asarray(m1, dtype=float)
    m2_arr = np.asarray(m2, dtype=float)
    m_tot = m1_arr + m2_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = (m1_arr * m2_arr) / (m_tot**2)
    return np.where(np.isfinite(nu), nu, 0.0)


def lognormal_mass_pdf(m: np.ndarray | float, m_c: float, sigma: float) -> np.ndarray:
    """Log-normal mass PDF psi(m) with median/central mass m_c and width sigma."""
    if m_c <= 0:
        raise ValueError("m_c must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    m_arr = np.asarray(m, dtype=float)
    out = np.zeros_like(m_arr, dtype=float)
    mask = m_arr > 0
    if not np.any(mask):
        return out

    coeff = 1.0 / (m_arr[mask] * sigma * np.sqrt(2.0 * np.pi))
    arg = (np.log(m_arr[mask] / m_c) ** 2) / (2.0 * sigma**2)
    out[mask] = coeff * np.exp(-arg)
    return out


def lognormal_mean(m_c: float, sigma: float) -> float:
    """Return <m> for log-normal distribution."""
    return float(m_c * np.exp(0.5 * sigma**2))


def lognormal_second_moment(m_c: float, sigma: float) -> float:
    """Return <m^2> for log-normal distribution."""
    return float(m_c**2 * np.exp(2.0 * sigma**2))


def _hyperu(a: float, b: float, z: float) -> float:
    """Confluent hypergeometric U(a,b,z) with SciPy first, mpmath fallback."""
    if _SCIPY_SPECIAL_AVAILABLE:
        return float(scipy_hyperu(a, b, z))

    try:
        import mpmath as mp
        return float(mp.hyperu(a, b, z))
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Need scipy.special.hyperu or mpmath.hyperu to evaluate C(f_PBH)."
        ) from exc


def c_fitting_function(fpbh: float, moment_ratio: float, sigma_m: float = DEFAULT_SIGMA_M) -> float:
    """
    Return fitting function C(f_PBH) entering S1.

    C = f^2 * (<m^2>/<m>^2)/sigma_m^2 * { [Gamma(29/37)/sqrt(pi) * U(21/74,1/2,5f^2/(6sigma_m^2))]^(-74/21) - 1 }^(-1)
    """
    if fpbh <= 0:
        raise ValueError("fpbh must be positive")
    if moment_ratio <= 0:
        raise ValueError("moment_ratio must be positive")
    if sigma_m <= 0:
        raise ValueError("sigma_m must be positive")

    z = 5.0 * fpbh**2 / (6.0 * sigma_m**2)
    gamma_val = scipy_gamma(29.0 / 37.0) if _SCIPY_SPECIAL_AVAILABLE else math.gamma(29.0 / 37.0)
    u_val = _hyperu(21.0 / 74.0, 0.5, z)

    pref = (gamma_val / np.sqrt(np.pi)) * u_val
    denom_term = pref ** (-74.0 / 21.0) - 1.0
    if denom_term <= 0:
        raise ValueError("Unphysical denominator in C(f_PBH); check fpbh/sigma_m values")

    return float((fpbh**2) * moment_ratio / (sigma_m**2) / denom_term)


def nbar_y(m_total: np.ndarray | float, mean_mass: float, fpbh: float, sigma_m: float = DEFAULT_SIGMA_M) -> np.ndarray:
    """Return Nbar(y) ~= (M/<m>) * f_PBH/(f_PBH + sigma_m)."""
    if mean_mass <= 0:
        raise ValueError("mean_mass must be positive")
    if fpbh <= 0:
        raise ValueError("fpbh must be positive")
    if sigma_m <= 0:
        raise ValueError("sigma_m must be positive")

    m_total_arr = np.asarray(m_total, dtype=float)
    return (m_total_arr / mean_mass) * (fpbh / (fpbh + sigma_m))


def suppression_s1(
    m_total: np.ndarray | float,
    mean_mass: float,
    second_moment: float,
    fpbh: float,
    sigma_m: float = DEFAULT_SIGMA_M,
) -> np.ndarray:
    """Return suppression factor S1 from the fitted approximation."""
    moment_ratio = second_moment / (mean_mass**2)
    c_val = c_fitting_function(fpbh=fpbh, moment_ratio=moment_ratio, sigma_m=sigma_m)
    nbar_val = nbar_y(m_total=m_total, mean_mass=mean_mass, fpbh=fpbh, sigma_m=sigma_m)

    bracket = moment_ratio / (nbar_val + c_val) + (sigma_m**2) / (fpbh**2)
    return 1.42 * (bracket ** (-21.0 / 74.0)) * np.exp(-nbar_val)


def suppression_s2(t_over_t0: np.ndarray | float, fpbh: float) -> np.ndarray:
    """Return S2(z) = min(1, 9.6e-3 f_tilde^-0.65 exp(0.03 ln(f_tilde)^2))."""
    if fpbh <= 0:
        raise ValueError("fpbh must be positive")

    t_ratio = np.asarray(t_over_t0, dtype=float)
    if np.any(t_ratio <= 0):
        raise ValueError("t_over_t0 must be positive")

    f_tilde = (t_ratio**0.44) * fpbh
    core = 9.6e-3 * (f_tilde ** (-0.65)) * np.exp(0.03 * (np.log(f_tilde) ** 2))
    return np.minimum(1.0, core)


def suppression_total(
    m_total: np.ndarray | float,
    t_over_t0: np.ndarray | float,
    mean_mass: float,
    second_moment: float,
    fpbh: float,
    sigma_m: float = DEFAULT_SIGMA_M,
) -> np.ndarray:
    """Return total suppression S = S1 * S2."""
    s1 = suppression_s1(
        m_total=m_total,
        mean_mass=mean_mass,
        second_moment=second_moment,
        fpbh=fpbh,
        sigma_m=sigma_m,
    )
    s2 = suppression_s2(t_over_t0=t_over_t0, fpbh=fpbh)
    return s1 * s2


def spin_cutoff_heaviside(nu: np.ndarray | float, nu_threshold: float | None) -> np.ndarray:
    """Return Theta(nu - nu_threshold); if threshold None, returns 1."""
    nu_arr = np.asarray(nu, dtype=float)
    if nu_threshold is None:
        return np.ones_like(nu_arr)
    if np.isinf(nu_threshold):
        return np.zeros_like(nu_arr)
    return (nu_arr >= nu_threshold).astype(float)


def _spin_model_value(nu: float, model: Literal["matched", "isco"] = "matched") -> float:
    if model == "matched":
        if not _MATCHED_SPIN_AVAILABLE:
            raise ImportError(
                "matched_a_star unavailable: could not import from SpinDistFromMassDist.py"
            )
        return float(np.asarray(matched_a_star(np.array([nu])))[0])

    if model == "isco":
        result = solve_a_star_self_consistent(nu, prograde=True)
        return float(result["a_star"])

    raise ValueError(f"Unknown spin model '{model}'")


def nu_threshold_from_spin_threshold(
    a_star_threshold: float,
    model: Literal["matched", "isco"] = "matched",
    nu_min: float = 1e-6,
    nu_max: float = 0.25,
    iterations: int = 80,
) -> float:
    """
    Infer nu_th such that a_star_model(nu_th) ~= a_star_threshold via bisection.

    Returns:
    - 0.0 if threshold is below model minimum on [nu_min, nu_max]
    - inf if threshold is above model maximum on [nu_min, nu_max]
    """
    if not (0.0 <= nu_min < nu_max <= 0.25):
        raise ValueError("Require 0 <= nu_min < nu_max <= 0.25")

    f_lo = _spin_model_value(nu_min, model=model)
    f_hi = _spin_model_value(nu_max, model=model)

    if a_star_threshold <= f_lo:
        return 0.0
    if a_star_threshold > f_hi:
        return float("inf")

    lo, hi = nu_min, nu_max
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = _spin_model_value(mid, model=model)
        if f_mid < a_star_threshold:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def differential_rate_non_poisson(
    m1: np.ndarray | float,
    m2: np.ndarray | float,
    m_c: float,
    sigma: float,
    fpbh: float = 1.0,
    t_over_t0: float = 1.0,
    nu_threshold: float | None = None,
    sigma_m: float = DEFAULT_SIGMA_M,
    rate_prefactor: float = RATE_PREFactor_GPC3_YR,
) -> np.ndarray:
    """
    Compute dR_np/(dm1 dm2) in Gpc^-3 yr^-1 Msun^-2.
    """
    if t_over_t0 <= 0:
        raise ValueError("t_over_t0 must be positive")

    m1_arr = np.asarray(m1, dtype=float)
    m2_arr = np.asarray(m2, dtype=float)
    if np.any(m1_arr <= 0) or np.any(m2_arr <= 0):
        raise ValueError("m1 and m2 must be positive")

    m_tot = m1_arr + m2_arr
    nu = symmetric_mass_ratio(m1_arr, m2_arr)

    mean_m = lognormal_mean(m_c=m_c, sigma=sigma)
    second_m = lognormal_second_moment(m_c=m_c, sigma=sigma)

    if fpbh <= 0:
        raise ValueError("fpbh must be positive")
    fpbh_arr = np.full_like(m_tot, float(fpbh))

    psi1 = lognormal_mass_pdf(m1_arr, m_c=m_c, sigma=sigma)
    psi2 = lognormal_mass_pdf(m2_arr, m_c=m_c, sigma=sigma)

    suppress = suppression_total(
        m_total=m_tot,
        t_over_t0=t_over_t0,
        mean_mass=mean_m,
        second_moment=second_m,
        fpbh=float(np.asarray(fpbh_arr).flat[0]) if np.allclose(fpbh_arr, fpbh_arr.flat[0]) else 1.0,
        sigma_m=sigma_m,
    )

    if not np.allclose(fpbh_arr, fpbh_arr.flat[0]):
        suppress = np.zeros_like(m_tot)
        for index in np.ndindex(m_tot.shape):
            suppress[index] = suppression_total(
                m_total=m_tot[index],
                t_over_t0=t_over_t0,
                mean_mass=mean_m,
                second_moment=second_m,
                fpbh=float(fpbh_arr[index]),
                sigma_m=sigma_m,
            )

    with np.errstate(divide="ignore", invalid="ignore"):
        nu_factor = np.where(nu > 0, nu ** (-34.0 / 37.0), 0.0)

    spin_step = spin_cutoff_heaviside(nu, nu_threshold)

    rate = (
        rate_prefactor
        * (fpbh_arr ** (53.0 / 37.0))
        * (t_over_t0 ** (-34.0 / 37.0))
        * (m_tot ** (-32.0 / 37.0))
        * spin_step
        * nu_factor
        * suppress
        * (psi1 * psi2) / (mean_m**2)
    )

    rate = np.where(np.isfinite(rate), rate, 0.0)
    return rate


def rate_grid(
    m1_values: np.ndarray,
    m2_values: np.ndarray,
    **rate_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return meshgrid (m1,m2) and differential rate values."""
    m1_grid, m2_grid = np.meshgrid(m1_values, m2_values, indexing="xy")
    dr = differential_rate_non_poisson(m1_grid, m2_grid, **rate_kwargs)
    return m1_grid, m2_grid, dr


def plot_rate_heatmap(
    m1_values: np.ndarray,
    m2_values: np.ndarray,
    log10_scale: bool = True,
    filename: str | None = None,
    show: bool = True,
    **rate_kwargs,
):
    """Plot dR/(dm1 dm2) over a mass grid."""
    m1_grid, m2_grid, dr = rate_grid(m1_values=m1_values, m2_values=m2_values, **rate_kwargs)

    data = np.log10(np.clip(dr, 1e-300, None)) if log10_scale else dr
    label = r"$\log_{10}[\mathrm{d}R/(\mathrm{d}m_1\mathrm{d}m_2)]$" if log10_scale else r"$\mathrm{d}R/(\mathrm{d}m_1\mathrm{d}m_2)$"

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(m1_grid, m2_grid, data, shading="auto")
    cb = plt.colorbar(mesh, ax=ax)
    cb.set_label(label)
    ax.set_xlabel(r"$m_1\,[M_\odot]$")
    ax.set_ylabel(r"$m_2\,[M_\odot]$")
    ax.set_title("PBH differential merger rate")
    ax.grid(alpha=0.2)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def integrate_rate_over_square_grid(
    m_min: float,
    m_max: float,
    n_points: int,
    **rate_kwargs,
) -> float:
    """Approximate total rate over m1,m2 in [m_min,m_max]^2 via trapezoidal integration."""
    if m_min <= 0 or m_max <= m_min:
        raise ValueError("Require 0 < m_min < m_max")
    if n_points < 3:
        raise ValueError("n_points must be at least 3")

    m_vals = np.logspace(np.log10(m_min), np.log10(m_max), n_points)
    m1_grid, m2_grid = np.meshgrid(m_vals, m_vals, indexing="xy")
    dr = differential_rate_non_poisson(m1_grid, m2_grid, **rate_kwargs)

    inner = _trapezoid(dr, x=m_vals, axis=1)
    total = _trapezoid(inner, x=m_vals)
    return float(total)


def convert_rate_volume_unit(
    rate_value: np.ndarray | float,
    from_unit: Literal["gpc", "kpc"] = "gpc",
    to_unit: Literal["gpc", "kpc"] = "gpc",
    ) -> np.ndarray | float:
    """Convert rates between yr^-1 Gpc^-3 and yr^-1 kpc^-3."""
    rate_arr = np.asarray(rate_value, dtype=float)

    if from_unit == to_unit:
        converted = rate_arr
    elif from_unit == "gpc" and to_unit == "kpc":
        # 1 Gpc = 1e6 kpc => 1/Gpc^3 = 1e-18 /kpc^3
        converted = rate_arr * 1.0e-18
    elif from_unit == "kpc" and to_unit == "gpc":
        converted = rate_arr * 1.0e18
    else:
        raise ValueError("from_unit and to_unit must be 'gpc' or 'kpc'")

    if np.ndim(rate_arr) == 0:
        return float(converted)
    return converted


def total_rate_above_spin_threshold(
    m_min: float,
    m_max: float,
    n_points: int,
    m_c: float,
    sigma: float,
    a_star_threshold: float,
    spin_model: Literal["matched", "isco"] = "matched",
    fpbh: float = 1.0,
    t_over_t0: float = 1.0,
    sigma_m: float = DEFAULT_SIGMA_M,
    rate_prefactor: float = RATE_PREFactor_GPC3_YR,
    nu_min: float = 1e-6,
    nu_max: float = 0.25,
    volume_unit: Literal["gpc", "kpc"] = "gpc",
    return_nu_threshold: bool = False,
) -> float | tuple[float, float]:
    """
    Return total merger rate above a spin threshold.

    This computes `nu_threshold` from the requested spin model and integrates
    `dR/(dm1 dm2)` over a square mass region `[m_min, m_max]^2` with
    `Theta(nu - nu_threshold)` applied.

    Returns
    -------
    float
        Total rate in `yr^-1 volume_unit^-3`.
    tuple[float, float]
        `(rate, nu_threshold)` if `return_nu_threshold=True`.
    """
    if a_star_threshold < 0:
        raise ValueError("a_star_threshold must be non-negative")
    if volume_unit not in {"gpc", "kpc"}:
        raise ValueError("volume_unit must be 'gpc' or 'kpc'")

    nu_threshold = nu_threshold_from_spin_threshold(
        a_star_threshold=a_star_threshold,
        model=spin_model,
        nu_min=nu_min,
        nu_max=nu_max,
    )

    rate_gpc = integrate_rate_over_square_grid(
        m_min=m_min,
        m_max=m_max,
        n_points=n_points,
        m_c=m_c,
        sigma=sigma,
        fpbh=fpbh,
        t_over_t0=t_over_t0,
        nu_threshold=nu_threshold,
        sigma_m=sigma_m,
        rate_prefactor=rate_prefactor,
    )
    rate = convert_rate_volume_unit(rate_gpc, from_unit="gpc", to_unit=volume_unit)

    if return_nu_threshold:
        return rate, nu_threshold
    return rate


def plot_rate_histogram_vs_final_spin(
    m1_values: np.ndarray,
    m2_values: np.ndarray,
    bins: int | np.ndarray = 60,
    spin_model: Literal["matched", "isco"] = "matched",
    volume_unit: Literal["gpc", "kpc"] = "gpc",
    filename: str | None = None,
    show: bool = True,
    **rate_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Plot rate-weighted histogram with final spin on x-axis and merger rate on y-axis.

    This is the 1D analogue of the 2D mass-grid rate calculation:
    1) evaluate dR/(dm1 dm2) on the (m1,m2) grid with no spin cutoff,
    2) map each mass-pair cell to a final spin value,
    3) sum cell-integrated rates into spin bins.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (hist_rate, spin_bin_edges), where hist_rate has units yr^-1 volume_unit^-3.

    Notes
    -----
    - If `bins` is an int, NumPy auto-chooses equal-width bins over the sampled spin range.
    - If `bins` is an array, it is used as explicit bin edges
      (e.g. `np.arange(0.0, 1.0 + 0.1, 0.1)`).
    """
    if np.isscalar(bins) and int(bins) < 2:
        raise ValueError("bins must be at least 2")
    if volume_unit not in {"gpc", "kpc"}:
        raise ValueError("volume_unit must be 'gpc' or 'kpc'")

    m1_values = np.asarray(m1_values, dtype=float)
    m2_values = np.asarray(m2_values, dtype=float)
    if np.any(m1_values <= 0) or np.any(m2_values <= 0):
        raise ValueError("m1_values and m2_values must be positive")

    rate_kwargs = dict(rate_kwargs)
    rate_kwargs["nu_threshold"] = None

    m1_grid, m2_grid, dr = rate_grid(m1_values=m1_values, m2_values=m2_values, **rate_kwargs)
    nu_grid = symmetric_mass_ratio(m1_grid, m2_grid)

    if spin_model == "matched":
        if not _MATCHED_SPIN_AVAILABLE:
            raise ImportError("matched_a_star unavailable: could not import from SpinDistFromMassDist.py")
        a_final = matched_a_star(nu_grid)
    elif spin_model == "isco":
        a_final = np.zeros_like(nu_grid)
        for index in np.ndindex(nu_grid.shape):
            a_final[index] = _spin_model_value(float(nu_grid[index]), model="isco")
    else:
        raise ValueError("spin_model must be 'matched' or 'isco'")

    dm1 = np.gradient(m1_values)
    dm2 = np.gradient(m2_values)
    cell_area = np.outer(dm2, dm1)
    rate_cell = dr * cell_area

    rate_cell = convert_rate_volume_unit(rate_cell, from_unit="gpc", to_unit=volume_unit)
    hist_rate, spin_edges = np.histogram(a_final.ravel(), bins=bins, weights=rate_cell.ravel())

    spin_centers = 0.5 * (spin_edges[:-1] + spin_edges[1:])
    widths = np.diff(spin_edges)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        spin_centers,
        hist_rate,
        width=widths,
        align="center",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xlabel(r"Final spin $a_{*,f}$")
    ax.set_ylabel(rf"Merger rate [yr$^{{-1}}$ {volume_unit}$^{{-3}}$]")
    ax.set_title("Merger-rate histogram vs final spin (no spin cutoff)")
    ax.grid(alpha=0.25)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return hist_rate, spin_edges


if __name__ == "__main__":

    a_star_threshold = 0.5
    fpbh = 1
    m_c = 1e-5
    sigma = 0.5
    nu_th = nu_threshold_from_spin_threshold(a_star_threshold=a_star_threshold, model="matched")
    
    
    # Calculate integration limits

    # Auto-compute bounds to capture ~99% of the distribution
    m_min = m_c * np.exp(-5 * sigma)  # Slightly conservative
    m_max = m_c * np.exp(5 * sigma)
    print(f"\nm_c: {m_c}, sigma: {sigma}, upper integration limits: {m_max}, lower integration limit: {m_min}\n")

    # Example threshold from matched spin model
    try:
        print(f"nu_threshold(a*={a_star_threshold}) = {nu_th:.5f}")
        rate_val = total_rate_above_spin_threshold(
            m_min=m_min,
            m_max=m_max,
            n_points=150,
            m_c=m_c,
            sigma=sigma,
            a_star_threshold=a_star_threshold,
            spin_model="matched",
            fpbh=fpbh,
            t_over_t0=1.0,
            volume_unit="kpc",
        )
        print(f"Rate above a*={a_star_threshold}: {rate_val:.6e} kpc^-3 yr^-1")
    except Exception as exc:
        print(f"Could not compute matched-model nu threshold: {exc}")

    try:
        m_vals = np.logspace(np.log10(m_min), np.log10(m_max), 500)
        fixed_spin_bins = np.arange(0.0, 1.0 + 0.1, 0.1)
        plot_rate_histogram_vs_final_spin(
            m1_values=m_vals,
            m2_values=m_vals,
            bins=fixed_spin_bins,
            spin_model="matched",
            volume_unit="kpc",
            m_c=m_c,
            sigma=sigma,
            fpbh=fpbh,
            t_over_t0=1.0,
            filename=None,
            show=True,
        )
    except Exception as exc:
        print(f"Could not plot merger-rate histogram vs final spin: {exc}")
