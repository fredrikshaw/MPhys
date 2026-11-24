"""Prepare reproducible favorable-polytope subsets from a local KS Cornell catalog."""

from bisect import bisect_left
from pathlib import Path
import random
import tempfile


def _catalog_parts(catalog):
    if catalog.is_file():
        return [catalog]
    parts = sorted(catalog.rglob("*.parquet"))
    if not parts:
        raise ValueError(f"No Parquet files found in {catalog}.")
    return parts


def prepare_samples(plan, catalog: Path, output: Path):
    """Sample known favorable populations in one pass over local Parquet.

    Each plan task supplies h11, available and n_polys. The initial rank
    samples are uniform without replacement within each h11. The final
    counts must agree with the plan before publishing any output.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    catalog = Path(catalog).resolve()
    output = Path(output).resolve()
    tasks = {task["h11"]: task for task in plan["tasks"]}
    if len(tasks) != len(plan["tasks"]) or not tasks:
        raise ValueError("The sampling plan must specify unique Hodge numbers.")
    for h11, task in tasks.items():
        if not 1 <= h11 <= 491 or not 1 <= task["n_polys"] <= task["available"]:
            raise ValueError(f"Invalid polytope count in the plan for h11={h11}.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Choose an empty or new samples directory. {output} already exists.")

    parts = _catalog_parts(catalog)
    selected = {
        h11: sorted(
            random.Random(f"{plan['seed']}:{h11}").sample(range(task["available"]), task["n_polys"])
        )
        for h11, task in tasks.items()
    }
    seen = {h11: 0 for h11 in tasks}
    next_rank = dict(seen)
    vertex_chunks = {h11: [] for h11 in tasks}
    columns = ["h11", "fav_N", "vertices"]
    include = np.zeros(492, dtype=bool)
    include[list(tasks)] = True
    scanned = 0

    for part_no, path in enumerate(parts, start=1):
        parquet = pq.ParquetFile(path)
        if not set(columns).issubset(parquet.schema_arrow.names):
            raise ValueError(f"{path} must include the KS Cornell columns {', '.join(columns)}.")
        for batch in parquet.iter_batches(batch_size=131_072, columns=columns):
            hodge = batch.column("h11").to_numpy(zero_copy_only=False)
            favorable = batch.column("fav_N").to_numpy(zero_copy_only=False)
            if np.any((hodge < 1) | (hodge > 491)) or hodge.dtype.kind not in "ui":
                raise ValueError(f"Invalid h11 values in {path}.")
            if favorable.dtype.kind != "b":
                raise ValueError(f"Missing or invalid favorability flags in {path}.")
            matches = np.flatnonzero(favorable & include[hodge])
            if not len(matches):
                continue
            matches = matches[np.argsort(hodge[matches], kind="stable")]
            sorted_hodge = hodge[matches]
            boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_hodge)) + 1, len(matches)]

            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                h11 = int(sorted_hodge[start])
                previous = seen[h11]
                seen[h11] += int(stop - start)
                ranks = selected[h11]
                first = next_rank[h11]
                last = bisect_left(ranks, seen[h11], lo=first)
                next_rank[h11] = last
                if first < last:
                    indices = np.asarray(ranks[first:last]) - previous + start
                    vertices = batch.column("vertices").take(matches[indices])
                    if vertices.null_count:
                        raise ValueError(f"Missing vertices in {path} at h11={h11}.")
                    vertex_chunks[h11].append(vertices)
        scanned += parquet.metadata.num_rows
        print(f"Scanned {part_no}/{len(parts)} catalog files ({scanned:,} rows).", flush=True)

    differences = [
        f"h11={h11}: found {seen[h11]:,}, expected {task['available']:,}"
        for h11, task in tasks.items()
        if seen[h11] != task["available"]
    ]
    if differences:
        text = "; ".join(differences[:5])
        raise ValueError(f"The catalog disagrees with the planned favorable counts ({text}).")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        staged = Path(tmp) / "samples"
        staged.mkdir()
        for h11, task in sorted(tasks.items()):
            vertices = pa.chunked_array(vertex_chunks[h11])
            count = task["n_polys"]
            if len(vertices) != count:
                raise RuntimeError(f"The prepared sample for h11={h11} is incomplete.")
            table = pa.table({
                "h11": pa.array([h11] * count, type=pa.int16()),
                "fav_N": pa.array([True] * count, type=pa.bool_()),
                "vertices": vertices,
            })
            pq.write_table(table, staged / f"h11_{h11}.parquet", compression="zstd")
        staged.rename(output)
    return output
