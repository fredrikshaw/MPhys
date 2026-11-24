"""Run independent mass calculations with restartable local checkpoints."""

from contextlib import closing
import multiprocessing as mp
from pathlib import Path
import sqlite3

import numpy as np
from tqdm import tqdm


_worker = None


def _set_worker(worker):
    global _worker
    _worker = worker


def _run_job(job):
    geometry_id, payload = job
    try:
        masses = np.asarray(_worker(payload), dtype="<f8")
        if masses.ndim != 1:
            raise ValueError("Expected a one-dimensional mass array.")
        return geometry_id, masses.tobytes(), None
    except Exception as exc:
        return geometry_id, None, f"{type(exc).__name__}: {exc}"


def _open_checkpoint(connection, run_signature):
    with connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS run "
            "(id INTEGER PRIMARY KEY CHECK (id = 1), signature TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS masses "
            "(geometry_id TEXT PRIMARY KEY, values_blob BLOB NOT NULL)"
        )
        connection.execute(
            "INSERT OR IGNORE INTO run (id, signature) VALUES (1, ?)",
            (run_signature,),
        )
        signature, = connection.execute("SELECT signature FROM run WHERE id = 1").fetchone()
        if signature != run_signature:
            raise ValueError("Checkpoint belongs to a different run. Use a new checkpoint path.")


def _save_results(connection, records):
    with connection:
        connection.executemany(
            "INSERT INTO masses (geometry_id, values_blob) VALUES (?, ?) "
            "ON CONFLICT(geometry_id) DO NOTHING",
            records,
        )
    records.clear()


def _collect_results(connection, results, *, total, completed, checkpoint_every):
    records = []
    failure_count = 0
    try:
        for geometry_id, values_blob, error in tqdm(
            results, total=total, initial=completed, desc="Processing polytopes"
        ):
            if error is not None:
                failure_count += 1
                tqdm.write(f"{geometry_id} failed ({error})")
                continue
            records.append((geometry_id, values_blob))
            if len(records) >= checkpoint_every:
                _save_results(connection, records)
    finally:
        if records:
            _save_results(connection, records)
    return failure_count


def run_mass_jobs(
    jobs, worker, *, workers, chunksize, checkpoint_path, run_signature, checkpoint_every=100
):
    """Return mass arrays in job order, resuming from successful checkpoints.

    Each job is a (unique geometry ID, worker input) tuple. The signature must
    identify the sample and every setting affecting its masses. Failed jobs
    remain in the queue for the next invocation.
    """
    if min(workers, chunksize, checkpoint_every) < 1:
        raise ValueError("Worker count, chunk size and checkpoint interval must be positive.")
    jobs = list(jobs)
    job_ids = [geometry_id for geometry_id, _ in jobs]
    if len(set(job_ids)) != len(jobs):
        raise ValueError("Geometry IDs must be unique within a run.")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(checkpoint_path)) as connection:
        _open_checkpoint(connection, run_signature)
        completed_ids = {row[0] for row in connection.execute("SELECT geometry_id FROM masses")}
        pending = [job for job in jobs if job[0] not in completed_ids]
        print(f"Resuming {len(jobs) - len(pending)} of {len(jobs)} polytopes from {checkpoint_path}")

        if pending:
            progress = {
                "total": len(jobs),
                "completed": len(jobs) - len(pending),
                "checkpoint_every": checkpoint_every,
            }
            if workers == 1 or len(pending) == 1:
                _set_worker(worker)
                failures = _collect_results(connection, map(_run_job, pending), **progress)
            else:
                pool_size = min(workers, len(pending))
                # Leave several chunks per worker so uneven solve times can rebalance.
                batch_size = min(chunksize, max(1, len(pending) // (4 * pool_size)))
                with mp.get_context("spawn").Pool(
                    processes=pool_size, initializer=_set_worker, initargs=(worker,)
                ) as pool:
                    results = pool.imap_unordered(_run_job, pending, chunksize=batch_size)
                    failures = _collect_results(connection, results, **progress)
            if failures:
                raise RuntimeError(
                    f"{failures} polytope calculation(s) failed. "
                    "Completed polytopes are saved. Rerun with the same settings to retry failures."
                )

        masses_by_id = dict(connection.execute("SELECT geometry_id, values_blob FROM masses"))
        return [np.frombuffer(masses_by_id[geometry_id], dtype="<f8").copy() for geometry_id in job_ids]
