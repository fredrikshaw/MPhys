"""Run a modest h11 sweep and compare the mass spectra on common log bins."""

import argparse
import json
import os
from pathlib import Path
import pickle
import sys

if __name__ == "__main__":
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")

import numpy as np

DEFAULT_H11 = (10, 30)


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h11", nargs="+", type=int, default=DEFAULT_H11)
    parser.add_argument("--n-polys", type=int, default=100, help="Maximum polytopes per h11.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--precision-dps", type=int, metavar="DIGITS",
                        help="Build and diagonalize the Hessian in multiprecision.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--online-sampling", choices=("full", "prefix"), default="full",
        help="Draw from the full online catalog or reproduce the legacy prefix sampling.",
    )
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--backend", choices=("topcom", "cgal", "qhull"), default="topcom")
    parser.add_argument("--source-file", type=Path, help="Local Kreuzer–Skarke text or KS Cornell Parquet.")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "massdist")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "mass_dist_pickles" / "pilot",
    )
    parser.add_argument(
        "--log-range", nargs=2, type=float, default=(-33, float(np.log10(2.435e27))),
        metavar=("MIN", "MAX"), help="Log-mass range used to calculate the plotted density.",
    )
    parser.add_argument(
        "--xlim", nargs=2, type=float, default=(-33, 30),
        metavar=("MIN", "MAX"), help="Displayed log-mass axis bounds.",
    )
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument(
        "--smooth-dex", type=float, default=0,
        help="Gaussian smoothing width in log10 mass decades. Zero draws a step histogram.",
    )
    parser.add_argument(
        "--plot-only", type=Path, metavar="SWEEP_SUMMARY",
        help="Replot existing pickles using the settings in sweep_summary.json.",
    )
    args = parser.parse_args(argv)
    if len(set(args.h11)) != len(args.h11) or any(not 1 <= value <= 491 for value in args.h11):
        parser.error("--h11 must contain distinct integers between 1 and 491.")
    if min(args.n_polys, args.workers, args.bins) < 1:
        parser.error("Polytope count, worker count and bin count must be positive.")
    if args.precision_dps is not None and args.precision_dps < 16:
        parser.error("--precision-dps must be at least 16.")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295.")
    if not np.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be finite and positive.")
    if not np.isfinite(args.log_range).all() or args.log_range[0] >= args.log_range[1]:
        parser.error("--log-range requires two increasing finite values.")
    if not np.isfinite(args.xlim).all() or args.xlim[0] >= args.xlim[1]:
        parser.error("--xlim requires two increasing finite values.")
    if not np.isfinite(args.smooth_dex) or args.smooth_dex < 0:
        parser.error("--smooth-dex must be finite and nonnegative.")
    return args


def _positive_log_masses(path):
    with path.open("rb") as handle:
        masses = np.asarray(pickle.load(handle))
    positive = masses[np.isfinite(masses) & (masses > 0)]
    return np.log10(positive), len(masses)


def _mass_density(log_masses, edges, smooth_dex):
    if smooth_dex:
        from scipy.ndimage import gaussian_filter1d

        # Use fine bins only as a numerical grid for smoothing in log mass.
        edges = np.linspace(edges[0], edges[-1], max(1201, 12 * len(edges)))
    counts, _ = np.histogram(log_masses, bins=edges)
    total = counts.sum()
    if not total:
        return edges, None
    density = counts / (total * np.diff(edges))
    if smooth_dex:
        spacing = edges[1] - edges[0]
        density = gaussian_filter1d(density, smooth_dex / spacing, mode="reflect")
        density /= np.dot(density, np.diff(edges))
    return edges, density


def _plot_spectra(samples, edges, scale, path, smooth_dex, xlim):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.3, 5.2), layout="constrained")
    for h11, n_polytopes, log_masses in samples:
        plot_edges, density = _mass_density(log_masses, edges, smooth_dex)
        if density is None:
            continue
        label = f"{h11} ({n_polytopes} polytopes)"
        if smooth_dex:
            centers = 0.5 * (plot_edges[:-1] + plot_edges[1:])
            ax.plot(centers, density, linewidth=1.6, label=label)
        else:
            ax.stairs(density, plot_edges, linewidth=1.6, label=label)
    if not ax.get_legend_handles_labels()[0]:
        plt.close(fig)
        return False
    if smooth_dex:
        fig.suptitle(f"Gaussian smoothing width: {smooth_dex:g} dex", fontsize=10)
    ax.set(
        xlabel=r'$\log_{10}(m_a/\mathrm{eV})$',
        ylabel="Probability density per decade",
        xlim=xlim,
        ylim=(0, 0.08),
        title=f"Axion mass spectra at the toric cone tip (stretch {scale:g})",
    )
    ax.legend(title=r"$h^{1,1}$ and sample size", frameon=False)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _save_plot(samples, edges, scale, output, smooth_dex, xlim):
    suffix = f"_smooth_{smooth_dex:g}dex" if smooth_dex else ""
    image = output / f"mass_spectra{suffix}.png"
    if _plot_spectra(samples, edges, scale, image, smooth_dex, xlim):
        print(f"Saved mass spectra to {image}", flush=True)
        return True
    print("No positive masses inside the plot range.", file=sys.stderr, flush=True)
    return False


def main(argv=None):
    args = _arguments(argv)
    if args.plot_only:
        output = args.plot_only.parent
        summary = json.loads(args.plot_only.read_text(encoding="utf-8"))
        samples = [
            (run["h11"], run["polytopes"], _positive_log_masses(output / run["pickle"])[0])
            for run in summary["runs"] if run["status"] == "completed"
        ]
        edges = np.asarray(summary["log10_mass_bin_edges"])
        return int(not _save_plot(samples, edges, summary["cone_scale"], output, args.smooth_dex, args.xlim))

    from MassDist import main as run_mass_distribution

    h11_values = sorted(args.h11)
    grid = "-".join(map(str, h11_values))
    settings = f"h11_{grid}_n{args.n_polys}_scale{args.scale:g}_{args.backend}_seed{args.seed}"
    if args.precision_dps is not None:
        settings += f"_dps{args.precision_dps}"
    if args.source_file is None and args.online_sampling == "full":
        settings += "_full_catalog"
    output = args.output_dir / settings
    output.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(*args.log_range, args.bins + 1)
    samples = []
    records = []

    for h11 in h11_values:
        print(f"\n=== h11 = {h11} ===", flush=True)
        command = [
            "--h11", str(h11), "--n-polys", str(args.n_polys), "--workers", str(args.workers),
            "--seed", str(args.seed), "--scale", str(args.scale), "--backend", args.backend,
            "--online-sampling", args.online_sampling,
            "--cache-dir", str(args.cache_dir), "--output-dir", str(output), "--no-plot",
        ]
        if args.precision_dps is not None:
            command.extend(("--precision-dps", str(args.precision_dps)))
        if args.source_file is not None:
            command.extend(("--source-file", str(args.source_file)))
        try:
            run = run_mass_distribution(command)
        except (Exception, SystemExit) as exc:
            print(f"h11={h11} failed: {exc}", file=sys.stderr, flush=True)
            records.append({"h11": h11, "status": "failed", "error": str(exc)})
            continue

        log_masses, n_saved = _positive_log_masses(run.output)
        plotted = int(np.count_nonzero((log_masses >= edges[0]) & (log_masses <= edges[-1])))
        if plotted:
            samples.append((h11, run.n_polytopes, log_masses))
        records.append({
            "h11": h11,
            "status": "completed",
            "polytopes": run.n_polytopes,
            "geometries": run.n_geometries,
            "mass_modes": run.n_modes,
            "finite_nonnegative_masses": n_saved,
            "excluded_modes": run.n_modes - n_saved,
            "positive_masses": len(log_masses),
            "positive_masses_in_plot": plotted,
            "pickle": run.output.name,
        })

    summary = {
        "polytopes_requested_per_h11": args.n_polys,
        "cone": "individual stretched toric cone",
        "cone_scale": args.scale,
        "triangulation_backend": args.backend,
        "triangulations_per_polytope": 1,
        "seed": args.seed,
        "precision_dps": args.precision_dps,
        "source_file": str(args.source_file.resolve()) if args.source_file else None,
        "online_sampling": args.online_sampling if args.source_file is None else None,
        "log10_mass_bin_edges": edges.tolist(),
        "runs": records,
    }
    summary_file = output / "sweep_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved sweep summary to {summary_file}", flush=True)
    plotted = _save_plot(samples, edges, args.scale, output, args.smooth_dex, args.xlim)
    return int(not plotted or any(record["status"] == "failed" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
