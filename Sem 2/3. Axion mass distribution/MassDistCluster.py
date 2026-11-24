"""Plan, prepare and distribute MassDist runs across Hodge numbers."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile


def _write_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(value)
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _write_json(path, value):
    _write_bytes(path, (json.dumps(value, indent=2) + "\n").encode())


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Expected an integer of at least 1.")
    return result


def _plan(args):
    if args.output.exists():
        raise ValueError(f"Plan already exists at {args.output}. Choose another filename.")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be between 0 and 4294967295.")
    if not math.isfinite(args.scale) or args.scale <= 0:
        raise ValueError("--scale must be finite and positive.")
    if args.precision_dps is not None and args.precision_dps < 16:
        raise ValueError("--precision-dps must be at least 16.")
    if args.backend == "topcom" and args.triangulations_per_polytope > 1:
        raise ValueError("Multiple triangulations require the CGAL or QHull backend.")

    counts = _read_json(args.counts)
    if args.h11 and any(not 1 <= h <= 491 for h in args.h11):
        raise ValueError("--h11 values must be between 1 and 491.")
    requested = sorted(set(args.h11 or range(1, 492)))
    available = {row["h11"]: row["favorable_polytopes_N"] for row in counts["counts"]}
    tasks = [
        {"h11": h, "available": available[h], "n_polys": min(available[h], args.cap)}
        for h in requested if available[h] > 0
    ]
    if not tasks:
        raise ValueError("The selected Hodge numbers have no favorable polytopes.")
    plan = {
        "format_version": 1,
        "count_source": counts.get("source"),
        "count_source_revision": counts.get("source_revision"),
        "seed": args.seed,
        "cap_per_h11": args.cap,
        "cone_scale": args.scale,
        "triangulation_backend": args.backend,
        "triangulations_per_polytope": args.triangulations_per_polytope,
        "precision_dps": args.precision_dps,
        "no_favorable_polytopes": [h for h in requested if available[h] == 0],
        "tasks": tasks,
    }
    _write_json(args.output, plan)
    print(f"Saved {len(tasks):,} tasks with {sum(t['n_polys'] for t in tasks):,} sampled polytopes to {args.output}")
    print(f"Task indices range from 0 to {len(tasks) - 1}.")


def _plan_id(plan):
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _prepare(args, plan):
    from MassDistCatalog import prepare_samples

    prepare_samples(plan, args.catalog, args.samples)
    _write_json(args.samples / "sample_plan.json", {"plan_id": _plan_id(plan)})
    print(f"Saved prepared samples to {args.samples}")


def _cpu_limit():
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    for variable in ("SLURM_CPUS_PER_TASK", "NSLOTS", "PBS_NP"):
        value = os.environ.get(variable, "")
        if value.isdigit() and int(value) > 0:
            available = min(available, int(value))
    return available


def _set_single_thread():
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS",
    ):
        os.environ[variable] = "1"


def _tasks(plan, batch_size):
    if batch_size is None:
        return plan["tasks"]
    tasks = []
    for task in plan["tasks"]:
        for start in range(0, task["n_polys"], batch_size):
            count = min(batch_size, task["n_polys"] - start)
            tasks.append({**task, "start": start, "count": count})
    return sorted(tasks, key=lambda task: _task_cost(task, plan), reverse=True)


def _task_cost(task, plan):
    power = 3 if plan["precision_dps"] is not None else 2
    return task.get("count", task["n_polys"]) * task["h11"]**power


def _run_root(args, plan):
    return args.run_dir or args.plan.with_suffix("") / _plan_id(plan)


def _output_dir(args, plan):
    output = _run_root(args, plan)
    return output if args.batch_size is None else output / f"blocks_{args.batch_size}"


def _task_label(task):
    label = f"h11_{task['h11']}"
    if "start" in task:
        label += f"_{task['start']}-{task['start'] + task['count']}"
    return label


def _task_report(output, task):
    return output / f"{_task_label(task)}.json"


def _run_task(args, plan, task, workers):
    _set_single_thread()
    from MassDist import main as run_mass_distribution

    h11 = task["h11"]
    source = args.samples / f"h11_{h11}.parquet"
    if not source.is_file():
        raise ValueError(f"Prepared sample for h11={h11} is missing at {source}.")
    output = _output_dir(args, plan)
    precision = plan["precision_dps"]
    command = [
        "--h11", str(h11), "--n-polys", str(task["n_polys"]), "--workers", str(workers),
        "--seed", str(plan["seed"]), "--scale", str(plan["cone_scale"]),
        "--backend", plan["triangulation_backend"],
        "--triangulations-per-polytope", str(plan["triangulations_per_polytope"]),
        "--source-file", str(source), "--cache-dir", str(output / "cache"),
        "--output-dir", str(output), "--no-plot",
    ]
    count = task.get("count", task["n_polys"])
    if "start" in task:
        command.extend(("--start-polytope", str(task["start"]), "--take-polys", str(count)))
    if precision is not None:
        command.extend(("--precision-dps", str(precision)))
    run = run_mass_distribution(command)
    if run.n_polytopes != count:
        raise ValueError(f"h11={h11} returned {run.n_polytopes} polytopes, expected {count}.")
    _write_json(_task_report(output, task), {
        "plan_id": _plan_id(plan),
        "h11": h11,
        "status": "completed",
        "polytopes": run.n_polytopes,
        "geometries": run.n_geometries,
        "mass_modes": run.n_modes,
        "sample_start": task.get("start", 0),
        "pickle": run.output.name,
    })


def _run_local(args, plan, tasks, workers):
    output = _output_dir(args, plan)
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()

    def run(index):
        label = _task_label(tasks[index])
        log = logs / f"{label}.log"
        command = [
            sys.executable, str(script), "run", str(args.plan), "--samples", str(args.samples),
            "--run-dir", str(_run_root(args, plan)), "--task-index", str(index), "--workers", str(workers),
        ]
        if args.batch_size is not None:
            command.extend(("--batch-size", str(args.batch_size)))
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        return label, result.returncode, log

    # Start expensive tasks early to reduce the tail at the end of the sweep.
    indices = sorted(
        range(len(tasks)), key=lambda i: _task_cost(tasks[i], plan), reverse=True,
    )
    failed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(run, index) for index in indices]):
            label, returncode, log = future.result()
            failed += bool(returncode)
            print(f"{label} {'failed; see' if returncode else 'completed; log at'} {log}", flush=True)
    return int(failed > 0)


def _collect_h11(output, h11, tasks, plan_id):
    reports = []
    for task in sorted(tasks, key=lambda task: task.get("start", 0)):
        path = _task_report(output, task)
        if not path.is_file():
            continue
        run = _read_json(path)
        if run.get("plan_id") != plan_id or not (output / run["pickle"]).is_file():
            continue
        if "start" in task and (
            run.get("sample_start") != task["start"] or run["polytopes"] != task["count"]
        ):
            continue
        reports.append(run)
    if len(reports) != len(tasks):
        return {"h11": h11, "status": "missing", "completed_blocks": len(reports), "blocks": len(tasks)}
    if len(reports) == 1:
        return reports[0]

    import numpy as np

    arrays = []
    for run in reports:
        with (output / run["pickle"]).open("rb") as stream:
            arrays.append(pickle.load(stream))
    name = f"masses_h11_{h11}_merged.pkl"
    _write_bytes(output / name, pickle.dumps(np.concatenate(arrays), protocol=pickle.HIGHEST_PROTOCOL))
    return {
        "plan_id": plan_id,
        "h11": h11,
        "status": "completed",
        "polytopes": sum(run["polytopes"] for run in reports),
        "geometries": sum(run["geometries"] for run in reports),
        "mass_modes": sum(run["mass_modes"] for run in reports),
        "blocks": len(reports),
        "pickle": name,
    }


def _collect(args, plan, tasks):
    output = _output_dir(args, plan)
    wanted = set(args.h11) if args.h11 else {task["h11"] for task in plan["tasks"]}
    unknown = wanted - {task["h11"] for task in plan["tasks"]}
    if unknown:
        raise ValueError(f"Hodge numbers absent from the plan: {sorted(unknown)}.")
    by_h11 = {h11: [] for h11 in wanted}
    for task in tasks:
        if task["h11"] in wanted:
            by_h11[task["h11"]].append(task)
    records = [_collect_h11(output, h11, by_h11[h11], _plan_id(plan)) for h11 in sorted(wanted)]
    start, stop = -33, math.log10(2.435e27)
    summary = {
        "cone": "individual stretched toric cone",
        "cone_scale": plan["cone_scale"],
        "precision_dps": plan["precision_dps"],
        "triangulation_backend": plan["triangulation_backend"],
        "triangulations_per_polytope": plan["triangulations_per_polytope"],
        "log10_mass_bin_edges": [start + (stop - start) * i / 60 for i in range(61)],
        "runs": records,
    }
    name = "sweep_summary.json" if not args.h11 else "sweep_summary_" + "-".join(map(str, sorted(wanted))) + ".json"
    path = output / name
    _write_json(path, summary)
    completed = sum(run["status"] == "completed" for run in records)
    print(f"Collected {completed} of {len(records)} tasks into {path}.")
    return int(completed != len(records))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    plan = commands.add_parser("plan", help="Create one task per Hodge number with favorable polytopes.")
    plan.add_argument("counts", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--h11", type=int, nargs="+")
    plan.add_argument("--cap", type=_positive_int, default=100)
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--scale", type=float, default=1)
    plan.add_argument("--backend", choices=("topcom", "cgal", "qhull"), default="topcom")
    plan.add_argument("--triangulations-per-polytope", type=_positive_int, default=1)
    plan.add_argument("--precision-dps", type=_positive_int)

    prepare = commands.add_parser("prepare", help="Select reproducible samples from a local KS Cornell snapshot.")
    prepare.add_argument("plan", type=Path)
    prepare.add_argument("--catalog", type=Path, required=True)
    prepare.add_argument("--samples", type=Path, required=True)

    run = commands.add_parser("run", help="Run one scheduler task or all tasks on one allocated node.")
    run.add_argument("plan", type=Path)
    run.add_argument("--samples", type=Path, required=True)
    run.add_argument("--task-index", type=int, help="Zero-based index. Defaults to SLURM_ARRAY_TASK_ID when set.")
    run.add_argument("--all", action="store_true")
    run.add_argument("--workers", type=_positive_int)
    run.add_argument("--jobs", type=_positive_int, default=1, help="Concurrent jobs with --all.")
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--batch-size", type=_positive_int, help="Maximum polytopes per independent job.")

    collect = commands.add_parser("collect", help="Combine completed task reports into a plot-ready summary.")
    collect.add_argument("plan", type=Path)
    collect.add_argument("--run-dir", type=Path)
    collect.add_argument("--batch-size", type=_positive_int, help="Use the same batch size as run.")
    collect.add_argument("--h11", type=int, nargs="+", help="Include only these Hodge numbers in the summary.")
    count = commands.add_parser("task-count", help="Print the number of tasks for a job array.")
    count.add_argument("plan", type=Path)
    count.add_argument("--batch-size", type=_positive_int)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.action == "plan":
            _plan(args)
            return 0
        plan = _read_json(args.plan)
        if plan["format_version"] != 1:
            raise ValueError("Unsupported plan version.")
        if args.action == "prepare":
            _prepare(args, plan)
            return 0
        tasks = _tasks(plan, args.batch_size)
        if args.action == "task-count":
            print(len(tasks))
            return 0
        if args.action == "collect":
            return _collect(args, plan, tasks)

        marker = _read_json(args.samples / "sample_plan.json")
        if marker.get("plan_id") != _plan_id(plan):
            raise ValueError("The prepared samples belong to a different plan.")
        cpus = _cpu_limit()
        parallel_jobs = args.jobs if args.all else 1
        workers = args.workers or max(1, min(8, cpus // parallel_jobs))
        if parallel_jobs * workers > cpus:
            raise ValueError(f"{parallel_jobs} jobs × {workers} workers exceeds the {cpus} CPUs available.")
        if args.all:
            if args.task_index is not None:
                raise ValueError("Use either --all or --task-index.")
            return _run_local(args, plan, tasks, workers)
        index = args.task_index
        slurm_index = os.environ.get("SLURM_ARRAY_TASK_ID", "")
        if index is None and slurm_index.isdigit():
            index = int(slurm_index)
        if index is None or not 0 <= index < len(tasks):
            raise ValueError(f"Choose --all or --task-index between 0 and {len(tasks) - 1}.")
        _run_task(args, plan, tasks[index], workers)
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
