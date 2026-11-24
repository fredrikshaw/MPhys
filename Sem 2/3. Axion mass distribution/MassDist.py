"""Compute and save axion mass spectra for favorable Kreuzer–Skarke polytopes."""

import argparse
from functools import partial
import hashlib
from importlib.metadata import version
from itertools import permutations
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import tempfile
from typing import NamedTuple, Sequence
import warnings

if __name__ == "__main__":
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")

from cytools import Polytope
import numpy as np

from MassDistBatch import run_mass_jobs
from MassDistInputs import load_polytope_vertices
from MassDistMath import AxionPotential, instanton_charges, multiprecision_masses

# Install the filter in the main process and in spawned workers.
warnings.filterwarnings(
    "ignore",
    message=r"(divide by zero|overflow|invalid value) encountered in det$",
    category=RuntimeWarning,
    module=r"numpy\.linalg(?:\..*)?$",
)

h11 = 10
n_polys = 100000
max_workers = 8  # Maximum number of polytopes to process concurrently
scaling_of_cone = 1

M_pl_in_ev = 1.22e28
W0 = 1.0

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

def axion_potential_optimized(theta: np.ndarray,
                              q: np.ndarray,
                              W0: float,
                              prefactor: float,
                              qa_dot_tau_precomp: np.ndarray,
                              exp_qa_dot_tau_precomp: np.ndarray,
                              qsum_dot_tau_precomp: np.ndarray,
                              exp_qsum_dot_tau_precomp: np.ndarray,
                              qa_ginv_qb_precomp: np.ndarray) -> float:
    """
    Optimized version of axion_potential with pre-computed expensive operations.
    Only theta-dependent terms are computed here.
    """
    theta = np.asarray(theta, dtype=float)
    n_inst = len(q)

    # First sum: only compute theta-dependent parts
    qa_dot_theta = q.dot(theta)
    term1 = np.sum(qa_dot_tau_precomp * W0 * exp_qa_dot_tau_precomp * np.cos(2*np.pi * qa_dot_theta))

    # Second double-sum: only compute theta-dependent parts
    term2 = 0.0
    idx = 0
    for a in range(n_inst):
        for b in range(a):
            qdiff_dot_theta = qa_dot_theta[a] - qa_dot_theta[b]
            bracket = qa_ginv_qb_precomp[idx] + qsum_dot_tau_precomp[idx]
            term2 += bracket * exp_qsum_dot_tau_precomp[idx] * np.cos(2*np.pi * qdiff_dot_theta)
            idx += 1

    V = prefactor * (term1 + term2)
    V_phys = V * M_pl_in_ev**4
    return float(V_phys)

def numerical_hessian(f, x0, eps=1e-5):
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    H = np.zeros((n, n))

    for i in range(n):
        dx_i = np.zeros(n)
        dx_i[i] = eps
        for j in range(i, n):
            dx_j = np.zeros(n)
            dx_j[j] = eps

            f_pp = f(x0 + dx_i + dx_j)
            f_pm = f(x0 + dx_i - dx_j)
            f_mp = f(x0 - dx_i + dx_j)
            f_mm = f(x0 - dx_i - dx_j)

            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
            if i != j:
                H[j, i] = H[i, j]

    return H


class Geometry(NamedTuple):
    q: np.ndarray
    tau: np.ndarray
    inverse_metric: np.ndarray
    volume: float
    precise_divisor_volumes: np.ndarray | None = None

    def scaled(self, cone_scale):
        # The base point is the unit-stretched tip on this same cone ray.
        divisor_volumes = self.precise_divisor_volumes
        if divisor_volumes is not None:
            divisor_volumes = divisor_volumes * cone_scale**2
        return Geometry(
            self.q,
            self.tau * cone_scale**2,
            self.inverse_metric * cone_scale**4,
            self.volume * cone_scale**3,
            divisor_volumes,
        )


def _precise_geometry(q, point, intersections):
    from mpmath import mp

    ctx = mp.clone()
    ctx.dps = 55
    t = ctx.matrix(point.tolist())
    contraction = ctx.matrix(len(point))
    for indices, coefficient in intersections.items():
        value = ctx.mpf(float(coefficient))
        for i, j, k in set(permutations(indices)):
            contraction[i, j] += value * t[k]
    tau = contraction * t / 2
    volume = (tau.T * t)[0] / 3
    # Convert the divisor volumes only after applying the charge matrix.
    # Rounding tau first would reintroduce cancellation in q @ tau.
    divisor_volumes = np.asarray((ctx.matrix(q.tolist()) * tau).tolist(), dtype=float).ravel()
    inverse_metric = 4 * (tau * tau.T - contraction * volume)
    return Geometry(q, np.asarray(tau.tolist(), dtype=float).ravel(),
                    np.asarray(inverse_metric.tolist(), dtype=float), float(volume), divisor_volumes)


def _geometry_at_tip(triangulation):
    cy = triangulation.get_cy()
    point = cy.toric_kahler_cone().tip_of_stretched_cone(1)
    if point is None:
        raise ValueError("CYTools did not find a stretched Kähler-cone tip.")
    # Contract CYTools' checked Calabi–Yau intersection numbers in the cone's
    # divisor basis. The fast fan volume helpers can give inaccurate
    # intersection coefficients, especially when high-h11 volumes cancel.
    kappa_point = np.zeros((len(point), len(point)))
    intersections = cy.intersection_numbers(in_basis=True)
    for indices, coefficient in intersections.items():
        for i, j, k in set(permutations(indices)):
            kappa_point[i, j] += coefficient * point[k]
    tau = (kappa_point @ point) / 2
    volume = float(tau @ point / 3)
    q = instanton_charges(cy)
    divisor_volumes = q @ tau
    if not np.all(np.isfinite(divisor_volumes) & (divisor_volumes > 0)):
        geometry = _precise_geometry(q, point, intersections)
        divisor_volumes = geometry.precise_divisor_volumes
        if not np.all(np.isfinite(divisor_volumes) & (divisor_volumes > 0)):
            raise ValueError(
                f"Invalid prime divisor volumes at the cone tip after a precision retry "
                f"(minimum {divisor_volumes.min():g})."
            )
        if not np.isfinite(geometry.volume) or geometry.volume <= 0:
            raise ValueError(f"Invalid Calabi–Yau volume at the cone tip ({geometry.volume:g}).")
        print("Used higher precision for divisor volumes at the original cone tip.", flush=True)
        return geometry
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError(f"Invalid Calabi–Yau volume at the cone tip ({volume:g}).")
    inverse_metric = 4 * (np.outer(tau, tau) - kappa_point * volume)
    return Geometry(q, tau, inverse_metric, volume)


def _potential(geometry):
    return AxionPotential(
        *geometry[:4], M_pl_in_ev, W0=W0, divisor_volumes=geometry.precise_divisor_volumes,
    )


def find_masses(
    polytope,
    silent=True,
    return_masses=True,
    return_potential=False,
    plot_hessian=False,
    plot_potential=False,
):
    """Compute masses or return the potential for a single TOPCOM geometry."""
    geometry = _geometry_at_tip(polytope.triangulate(backend="topcom")).scaled(scaling_of_cone)
    potential = _potential(geometry)
    masses = potential.masses() if return_masses else None

    if not silent:
        print(f"Charge matrix shape: {geometry.q.shape}")
        print(f"Charge matrix rank: {np.linalg.matrix_rank(geometry.q)}")
        if masses is not None:
            finite = masses[np.isfinite(masses)]
            print(f"Finite masses: {finite.size}/{masses.size}")
            if finite.size:
                print(f"Mass range [eV]: {finite.min():.6g} to {finite.max():.6g}")

    if plot_hessian or plot_potential:
        import matplotlib.pyplot as plt

        if plot_hessian:
            _, ax = plt.subplots()
            image = ax.imshow(potential.hessian(), cmap="RdBu_r")
            ax.set(xlabel="Axion index", ylabel="Axion index", title="Axion Hessian at the origin")
            plt.colorbar(image, label=r"Hessian element [eV$^4$]")
        if plot_potential:
            angles = np.linspace(-0.5, 0.5, 500)
            _, ax = plt.subplots()
            ax.plot(angles, [potential(angle * np.ones(len(geometry.tau))) for angle in angles])
            ax.set(xlabel="Common axion angle (period 1)", ylabel=r"Potential [eV$^4$]")
        plt.show()

    if return_masses and return_potential:
        return masses, potential
    if return_potential:
        return potential
    return masses


def _signature(*values):
    serialized = json.dumps(values, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()


def _write_atomic(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            write(handle)
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _compute_geometry_batch(vertices, task_seed, backend, triangulations):
    polytope = Polytope(vertices)
    if triangulations == 1:
        return [_geometry_at_tip(polytope.triangulate(backend=backend))]
    # A small polytope may have fewer than the requested number of distinct FRSTs.
    sample = polytope.random_triangulations_fast(
        N=triangulations, backend=backend, seed=task_seed, progress_bar=False,
    )
    geometries = [_geometry_at_tip(triangulation) for triangulation in sample]
    if not geometries:
        raise ValueError("CYTools did not find any fine, regular, star triangulations.")
    return geometries


def _load_geometry_batch(vertices, task_seed, backend, triangulations, geometry_cache, geometry_version):
    if geometry_cache is None:
        return _compute_geometry_batch(vertices, task_seed, backend, triangulations)

    sample_seed = task_seed if triangulations > 1 else None
    key = _signature(geometry_version, vertices, sample_seed, backend, triangulations)
    path = geometry_cache / key[:2] / f"{key}.npz"
    base_fields = Geometry._fields[:-1]
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            arrays = [data[field] for field in base_fields]
            return [
                Geometry(*values, data.get(f"precise_divisor_volumes_{index}"))
                for index, values in enumerate(zip(*arrays))
            ]

    geometries = _compute_geometry_batch(vertices, task_seed, backend, triangulations)
    arrays = {field: np.stack([getattr(geometry, field) for geometry in geometries])
              for field in base_fields}
    arrays.update({
        f"precise_divisor_volumes_{index}": geometry.precise_divisor_volumes
        for index, geometry in enumerate(geometries)
        if geometry.precise_divisor_volumes is not None
    })
    _write_atomic(path, lambda handle: np.savez(handle, **arrays))
    return geometries


def _find_masses_worker(
    job, *, scale, backend, triangulations, geometry_cache, geometry_version, precision_dps=None,
):
    vertices, task_seed = job
    geometries = _load_geometry_batch(
        vertices, task_seed, backend, triangulations, geometry_cache, geometry_version,
    )
    scaled = [geometry.scaled(scale) for geometry in geometries]
    if precision_dps is None:
        return np.concatenate([_potential(geometry).masses() for geometry in scaled])
    return np.concatenate([
        multiprecision_masses(
            *geometry[:4], M_pl_in_ev, W0=W0, dps=precision_dps,
            divisor_volumes=geometry.precise_divisor_volumes,
        )
        for geometry in scaled
    ])


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h11", type=int, default=h11)
    parser.add_argument("--n-polys", type=int, default=n_polys)
    parser.add_argument("--scale", type=float, default=scaling_of_cone)
    parser.add_argument("--workers", "--max-workers", type=int, default=max_workers)
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument(
        "--precision-dps", type=int, metavar="DIGITS",
        help="Build and diagonalize the Hessian with this many decimal digits; default is float64.",
    )
    parser.add_argument(
        "--start-polytope", type=int, default=0, metavar="INDEX",
        help="Start at this zero-based index in the full seeded sample.",
    )
    parser.add_argument(
        "--take-polys", type=int, metavar="COUNT",
        help="Calculate up to COUNT polytopes starting at --start-polytope in the seeded sample.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--online-sampling", choices=("full", "prefix"), default="full",
        help="Draw from the full online catalog or reproduce the legacy prefix sampling.",
    )
    parser.add_argument("--backend", choices=("topcom", "cgal", "qhull"), default="topcom")
    parser.add_argument(
        "--triangulations-per-polytope", type=int, default=1,
        help="Request up to this many distinct FRSTs using CYTools fast sampling with CGAL or QHull.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path.home() / ".cache" / "massdist",
        help="Local directory for cached vertex samples and resumable mass checkpoints.",
    )
    parser.add_argument(
        "--source-file", type=Path, help="Local Kreuzer–Skarke text or KS Cornell Parquet source."
    )
    parser.add_argument(
        "--cache-geometry", action="store_true",
        help="Save reusable geometry at the unit-stretched toric cone tip (adds disk usage).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "mass_dist_pickles",
    )
    parser.add_argument("--no-plot", action="store_true", help="Save masses without opening a plot.")
    args = parser.parse_args(argv)
    if not 1 <= args.h11 <= 491:
        parser.error("--h11 must be between 1 and 491.")
    if min(args.n_polys, args.workers, args.chunksize, args.triangulations_per_polytope) < 1:
        parser.error("Polytope count, worker count, chunk size and triangulation count must be positive.")
    if args.precision_dps is not None and args.precision_dps < 16:
        parser.error("--precision-dps must be at least 16.")
    if args.start_polytope < 0:
        parser.error("--start-polytope must be nonnegative.")
    if args.take_polys is not None and args.take_polys < 1:
        parser.error("--take-polys must be positive.")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295.")
    if not np.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be finite and positive.")
    if args.triangulations_per_polytope > 1 and args.backend == "topcom":
        parser.error("Multiple triangulations require --backend cgal or --backend qhull.")
    return args


class MassRun(NamedTuple):
    output: Path
    n_polytopes: int
    n_geometries: int
    n_modes: int


def main(argv=None):
    args = _arguments(argv)
    vertices = load_polytope_vertices(
        args.h11, args.n_polys, cache_dir=args.cache_dir / "vertices",
        seed=args.seed, source_file=args.source_file, online_sampling=args.online_sampling,
    )
    if not vertices:
        raise SystemExit(f"No favorable polytopes found for h11={args.h11}.")
    start = args.start_polytope
    if start >= len(vertices):
        raise ValueError(f"Start index {start} lies outside the sample of {len(vertices)} polytopes.")
    stop = None if args.take_polys is None else start + args.take_polys
    vertices = vertices[start:stop]
    # Keep geometry caches independent of cone scale and mass checkpoints.
    geometry_version = _signature("unit-cone-tip-checked-intersections-v2", version("cytools"), version("numpy"))
    geometry_cache = args.cache_dir / "geometry" if args.cache_geometry else None
    rng = np.random.default_rng(args.seed)
    # Advance the same seed stream for slices as for the unsliced sample.
    for _ in range(start):
        rng.integers(2**32)
    jobs = [
        (str(index), (coords, int(rng.integers(2**32))))
        for index, coords in enumerate(vertices, start=start)
    ]
    precision_key = ()
    precision_label = "float64"
    precision_suffix = ""
    if args.precision_dps is not None:
        precision_key = ("multiprecision-masses-v1", version("mpmath"), args.precision_dps)
        precision_label = f"{args.precision_dps} decimal digits"
        precision_suffix = f"_dps{args.precision_dps}"
    signature = _signature(
        "analytic-masses-v1", geometry_version, version("scipy"), M_pl_in_ev, W0,
        args.h11, args.scale, args.backend, args.triangulations_per_polytope, jobs,
        *precision_key,
    )
    workers = min(len(jobs), mp.cpu_count(), args.workers)
    print(f"Using {len(jobs):,} favorable polytopes (requested up to {args.n_polys:,})")
    print(f"Using {workers} worker(s), {args.backend} backend, cone scale {args.scale:g}")
    print(f"Hessian and eigensolver precision: {precision_label}")
    worker = partial(
        _find_masses_worker, scale=args.scale, backend=args.backend,
        triangulations=args.triangulations_per_polytope, geometry_cache=geometry_cache,
        geometry_version=geometry_version, precision_dps=args.precision_dps,
    )
    mass_list = run_mass_jobs(
        jobs, worker, workers=workers, chunksize=args.chunksize,
        checkpoint_path=args.cache_dir / "checkpoints" / f"{signature[:20]}.sqlite3",
        run_signature=signature,
    )
    if any(len(masses) % args.h11 for masses in mass_list):
        raise ValueError(f"Worker returned a spectrum with incompatible dimension for h11={args.h11}.")
    spectra = np.concatenate(mass_list)
    n_geometries = len(spectra) // args.h11
    all_masses = spectra[np.isfinite(spectra)]
    filename = (
        f"masses_h11_{args.h11}_npolys_{len(jobs)}_scaling_{args.scale:g}"
        f"_nmasses_{len(all_masses)}{precision_suffix}_{signature[:12]}.pkl"
    )
    output = args.output_dir / filename
    _write_atomic(output, lambda handle: pickle.dump(all_masses, handle, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"Saved {len(all_masses):,} finite masses from {n_geometries:,} geometries to {output}")
    excluded = len(spectra) - len(all_masses)
    if excluded:
        reason = "negative squared masses or nonfinite mass"
        if args.precision_dps is not None:
            reason = "negative or unresolved squared masses, or nonfinite mass"
        print(f"Excluded {excluded:,} modes with {reason}.")

    result = MassRun(output, len(jobs), n_geometries, len(spectra))
    if not args.no_plot:
        import matplotlib.pyplot as plt

        positive = all_masses[all_masses > 0]
        if not len(positive):
            print("No positive masses to plot on a logarithmic axis.")
            return result
        logs = np.log10(positive)
        _, ax = plt.subplots(figsize=(12, 7))
        ax.hist(logs, bins=50, density=True, color="steelblue", edgecolor="black")
        ax.set(xlabel=r"$\log_{10}(m_a/\mathrm{eV})$", ylabel="Density",
               xlim=(-33, 30),
               title=f"{n_geometries:,} geometries, h11={args.h11}, cone scale={args.scale:g}"
                     f" ({precision_label})")
        plt.show()
    return result


if __name__ == "__main__":
    main()
