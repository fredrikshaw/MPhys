"""Reproducible samples of favorable N-lattice polytopes for MassDist."""

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import tempfile


_KS_QUERY = "https://quark.itp.tuwien.ac.at/cgi-bin/cy/cydata.cgi"
_KS_HEADER = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+M:[^\n]*?\bH:\s*(\d+)\s*,\s*(\d+)\b"
)


def _ks_vertices(lines, h11):
    """Read KS vertex records in the same orientation as CYTools' N query."""
    lines = iter(lines)
    for line in lines:
        header = _KS_HEADER.match(line)
        if header is None:
            continue
        rows, columns, _, hodge_n = map(int, header.groups())
        coordinates = [list(map(int, next(lines, "").split())) for _ in range(rows)]
        if not all(len(row) == columns for row in coordinates):
            raise ValueError("Incomplete or invalid Kreuzer–Skarke vertex record.")
        if hodge_n == h11:
            yield [list(vertex) for vertex in zip(*coordinates)] if columns > rows else coordinates


def _sample_online(h11, limit, seed):
    import requests
    from cytools import Polytope

    # A random visit order gives a sample without replacement from the favorable
    # records in the retrieved prefix. A truncated prefix is not the catalog.
    fetch_limit = max(100, 2 * limit)
    previous_count = 0
    with requests.Session() as session:
        while True:
            print(f"Fetching up to {fetch_limit:,} Kreuzer–Skarke records for h11={h11}.", flush=True)
            response = session.get(_KS_QUERY, params={"h12": h11, "L": fetch_limit}, timeout=(15, 180))
            response.raise_for_status()
            body = response.text
            if "<b>Result:</b>" not in body or "</pre>" not in body.lower():
                raise ValueError("Kreuzer–Skarke returned an incomplete or unexpected response.")
            catalog = list(_ks_vertices(io.StringIO(body), h11))
            truncated = len(catalog) >= fetch_limit or "Exceeded limit" in body
            if not catalog and not re.search(r"#NF:\s*0\b", body):
                raise ValueError("Kreuzer–Skarke returned no readable polytope records.")
            order = list(range(len(catalog)))
            random.Random(seed).shuffle(order)
            sample = []
            for index in order:
                vertices = catalog[index]
                if Polytope(vertices).is_favorable(lattice="N"):
                    sample.append(vertices)
                    if len(sample) == limit:
                        break
            if len(sample) >= limit or not truncated:
                population = "downloaded catalog prefix" if truncated else "full query results"
                return sample, population
            if len(catalog) <= previous_count:
                raise ValueError(
                    "The database returned no additional polytopes at a larger limit. "
                    "Use a smaller sample or a local catalog with --source-file."
                )
            previous_count = len(catalog)
            fetch_limit *= 2


def _sample_local_ks(path, h11, limit, seed):
    from cytools import Polytope

    rng = random.Random(seed)
    sample = []
    favorable_count = 0
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for vertices in _ks_vertices(stream, h11):
            if not Polytope(vertices).is_favorable(lattice="N"):
                continue
            favorable_count += 1
            if len(sample) < limit:
                sample.append(vertices)
            else:
                index = rng.randrange(favorable_count)
                if index < limit:
                    sample[index] = vertices
    return sample, "local Kreuzer–Skarke records"


def _sample_parquet(paths, h11, limit, seed):
    import pyarrow as pa
    import pyarrow.dataset as ds

    dataset = ds.dataset([str(path) for path in paths], format="parquet")
    if not {"h11", "fav_N", "vertices"}.issubset(dataset.schema.names):
        raise ValueError("Parquet source must have the KS Cornell h11, fav_N, and vertices columns.")
    lookup = (ds.field("h11") == h11) & ds.field("fav_N")
    population = "favorable rows in the local KS Cornell Parquet source"
    count = dataset.count_rows(filter=lookup)
    if count == 0:
        return [], population
    indices = sorted(random.Random(seed).sample(range(count), min(count, limit)))
    sample = []
    offset = 0
    next_index = 0
    for batch in dataset.to_batches(columns=["vertices"], filter=lookup):
        stop = offset + batch.num_rows
        within_batch = []
        while next_index < len(indices) and indices[next_index] < stop:
            within_batch.append(indices[next_index] - offset)
            next_index += 1
        if within_batch:
            sample.extend(batch.column(0).take(pa.array(within_batch)).to_pylist())
        if next_index == len(indices):
            break
        offset = stop
    if len(sample) != len(indices):
        raise ValueError("Local Parquet source changed while calculating the sample.")
    return sample, population


def _write_cache(path, vertices, population):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8", compresslevel=6) as stream:
            json.dump({"population": population, "vertices": vertices}, stream, separators=(",", ":"))
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_polytope_vertices(h11, limit, *, cache_dir, seed=0, source_file=None, online_sampling="full"):
    """Load a seeded sample from its cache, KS online, or a local catalog.

    Local Parquet data must follow the KS Cornell schema and use the N-lattice
    orientation. KS text files use the original database's H:M,N header order.
    """
    if limit < 1:
        raise ValueError("The number of requested polytopes must be positive.")
    source = None if source_file is None else Path(source_file).resolve()
    paths = None
    if online_sampling not in ("full", "prefix"):
        raise ValueError("Online sampling must be full or prefix.")
    source_id = "ks_https"
    if source is None and online_sampling == "full":
        from MassDistRemote import REVISION, sample_full_catalog

        source_id = f"ks_cornell_full_{REVISION[:12]}"
    if source is not None:
        paths = sorted(source.rglob("*.parquet")) if source.is_dir() else [source]
        if not paths:
            raise ValueError(f"No Parquet files found in {source}.")
        file_identity = [
            (str(path), stat.st_size, stat.st_mtime_ns)
            for path in paths
            for stat in [path.stat()]
        ]
        source_id = hashlib.sha256(repr(file_identity).encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"vertices_v1_{source_id}_h11_{h11}_n{limit}_seed{seed}.json.gz"
    if cache.is_file():
        with gzip.open(cache, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        print(f"Loaded {len(data['vertices']):,} favorable polytopes from {cache}.", flush=True)
        print(f"Sampling population: {data['population']}.", flush=True)
        return data["vertices"]
    if source is None:
        if online_sampling == "full":
            vertices, population = sample_full_catalog(h11, limit, seed, cache_dir=cache_dir)
        else:
            vertices, population = _sample_online(h11, limit, seed)
    elif source.is_dir() or source.suffix.lower() == ".parquet":
        vertices, population = _sample_parquet(paths, h11, limit, seed)
    else:
        vertices, population = _sample_local_ks(source, h11, limit, seed)
    _write_cache(cache, vertices, population)
    print(f"Cached {len(vertices):,} favorable polytopes in {cache}.", flush=True)
    print(f"Sampling population: {population}.", flush=True)
    return vertices
