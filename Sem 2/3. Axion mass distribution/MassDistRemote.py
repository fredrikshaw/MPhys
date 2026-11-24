"""Sample favorable rows from a pinned KS Cornell snapshot without a local copy."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import random
import re
import sqlite3
import tempfile

import numpy as np
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from MassDistParquetHttp import RangeReader


CATALOG = "aschachner/KS_Cornell"
REVISION = "79879a2ab3d0159f0a4ce668a73ec550f8c9dc8e"
TOTAL_ROWS = 473_800_776
WORKERS = 6
VIEWER_SAMPLE_LIMIT = 24
HF = "https://huggingface.co"
ROWS = "https://datasets-server.huggingface.co/rows"


def _session(*, retry_rate_limit=True):
    session = requests.Session()
    statuses = (429, 500, 502, 503, 504) if retry_rate_limit else (500, 502, 503, 504)
    retry = Retry(total=3, status_forcelist=statuses, backoff_factor=1)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _parquet_files():
    tree_url = f"{HF}/api/datasets/{CATALOG}/tree"
    with _session() as session:
        response = session.get(f"{tree_url}/{REVISION}", timeout=(15, 60))
        response.raise_for_status()
        files = sorted(
            (item["path"], item["size"], item["lfs"]["oid"])
            for item in response.json()
            if item.get("type") == "file" and re.fullmatch(r"ks_\d+\.parquet", item["path"])
        )
    if len(files) != 48:
        raise ValueError("The pinned KS Cornell catalog did not list all 48 Parquet files.")
    return files


def _check_viewer_order(files, session):
    tree_url = f"{HF}/api/datasets/{CATALOG}/tree"
    response = session.get(
        f"{tree_url}/refs%2Fconvert%2Fparquet", params={"recursive": "true"}, timeout=(15, 60),
    )
    response.raise_for_status()
    viewer = sorted(
        (item["path"], item["size"], item["lfs"]["oid"])
        for item in response.json()
        if item.get("type") == "file" and re.fullmatch(r"default/train/\d+\.parquet", item["path"])
    )
    # Viewer row offsets only apply if its file order matches the pinned files.
    if [entry[1:] for entry in files] != [entry[1:] for entry in viewer]:
        raise ValueError("The KS Cornell row service no longer matches the pinned catalog files.")


def _parquet_url(name):
    return f"{HF}/datasets/{CATALOG}/resolve/{REVISION}/{name}"


def _favorable_offsets(item, h11):
    name, size, _ = item
    url = _parquet_url(name)
    groups = []
    offset = 0
    with RangeReader(url, size) as reader:
        source = pq.ParquetFile(reader, pre_buffer=False)
        columns = source.schema_arrow.names
        if "h11" not in columns or "fav_N" not in columns:
            raise ValueError(f"Missing Hodge number or favorability in {name}.")
        for index in range(source.num_row_groups):
            group = source.metadata.row_group(index)
            stats = next(
                (group.column(j).statistics for j in range(group.num_columns)
                 if group.column(j).path_in_schema == "h11"), None,
            )
            if stats is None or not stats.has_min_max or stats.min <= h11 <= stats.max:
                table = source.read_row_group(index, columns=["h11", "fav_N"], use_threads=False)
                hodge = table["h11"].combine_chunks()
                favorable = table["fav_N"].combine_chunks()
                if hodge.null_count or favorable.null_count:
                    raise ValueError(f"Incomplete Hodge number or favorability column in {name}.")
                matches = np.flatnonzero(
                    (hodge.to_numpy(zero_copy_only=False) == h11)
                    & favorable.to_numpy(zero_copy_only=False)
                )
                if matches.size:
                    groups.append((offset, matches.astype(np.int32)))
            offset += group.num_rows
        return offset, groups, reader.bytes_read


def _catalog_index(h11, cache_dir, files):
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"ks_cornell_{REVISION[:12]}_h11_{h11}_row_index.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                offsets = data["offsets"]
            print(f"Loaded the index for {len(offsets):,} favorable polytopes from disk.", flush=True)
            return offsets

    print(f"Indexing h11={h11} across all {len(files)} KS Cornell files.", flush=True)
    pieces = []
    file_offset = transferred = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for index, (rows_in_file, groups, size) in enumerate(
            pool.map(lambda item: _favorable_offsets(item, h11), files), 1
        ):
            pieces.extend(file_offset + start + matches for start, matches in groups)
            file_offset += rows_in_file
            transferred += size
            if index % 12 == 0:
                print(f"Indexed {index}/{len(files)} files.", flush=True)
    if file_offset != TOTAL_ROWS:
        raise ValueError("The KS Cornell catalog scan did not cover the expected number of rows.")
    offsets = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int32)
    print(f"Indexed {len(offsets):,} favorable polytopes ({transferred / 1e6:.0f} MB read).", flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                np.savez_compressed(handle, offsets=offsets)
                handle.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    return offsets


def _viewer_vertices(files, row_numbers, h11):
    with _session(retry_rate_limit=False) as session:
        _check_viewer_order(files, session)
        for row_number in row_numbers:
            response = session.get(
                ROWS,
                params={"dataset": CATALOG, "config": "default", "split": "train",
                        "offset": row_number, "length": 1},
                timeout=(15, 60),
            )
            response.raise_for_status()
            if response.headers.get("X-Revision") != REVISION:
                raise ValueError("The KS Cornell row service no longer matches the pinned catalog revision.")
            payload = response.json()
            rows = payload["rows"]
            if payload["num_rows_total"] != TOTAL_ROWS or payload.get("partial") is not False or len(rows) != 1:
                raise ValueError(f"The catalog row service returned an incomplete result for row {row_number}.")
            record = rows[0]
            values = record["row"]
            vertices = values.get("vertices")
            if (record["row_idx"] != row_number or values.get("h11") != h11
                    or values.get("fav_N") is not True or not vertices
                    or "vertices" in record.get("truncated_cells", [])):
                raise ValueError(f"The catalog row service could not verify favorable row {row_number}.")
            yield row_number, vertices


def _save_rows(connection, h11, rows):
    with connection:
        connection.executemany(
            "INSERT OR REPLACE INTO vertices (h11, row_number, coordinates) VALUES (?, ?, ?)",
            ((h11, number, json.dumps(coordinates, separators=(",", ":"))) for number, coordinates in rows),
        )


def _selected_vertices(files, row_numbers, h11, cache_dir):
    database = ":memory:"
    if cache_dir is not None:
        path = Path(cache_dir) / f"ks_cornell_{REVISION[:12]}_vertex_rows.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        database = path

    with closing(sqlite3.connect(database, timeout=60)) as connection:
        with connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS vertices ("
                "h11 INTEGER NOT NULL, row_number INTEGER NOT NULL, coordinates TEXT NOT NULL, "
                "PRIMARY KEY (h11, row_number))"
            )
        selected = set(row_numbers)
        vertices = {
            number: json.loads(coordinates)
            for number, coordinates in connection.execute(
                "SELECT row_number, coordinates FROM vertices WHERE h11 = ?", (h11,),
            )
            if number in selected
        }
        missing = [number for number in row_numbers if number not in vertices]
        if vertices:
            print(f"Reusing {len(vertices):,}/{len(row_numbers):,} downloaded vertex lists.", flush=True)

        if 0 < len(missing) <= VIEWER_SAMPLE_LIMIT:
            try:
                for number, coordinates in _viewer_vertices(files, missing, h11):
                    _save_rows(connection, h11, [(number, coordinates)])
                    vertices[number] = coordinates
            except (OSError, ValueError, KeyError):
                print("The row viewer is unavailable. Reading the remaining rows from Parquet.", flush=True)
            missing = [number for number in missing if number not in vertices]

        if missing:
            from MassDistParquetRows import iter_parquet_vertices

            sources = [(_parquet_url(name), size) for name, size, _ in files]
            # These offsets were already filtered by h11 and favorability in
            # this pinned revision. Read just the vertex column in this pass.
            transferred = 0
            last_update = len(vertices)
            for rows, size in iter_parquet_vertices(
                sources, missing, h11, total_rows=TOTAL_ROWS, workers=WORKERS, verify=False,
            ):
                _save_rows(connection, h11, rows)
                vertices.update(rows)
                transferred += size
                if len(vertices) - last_update >= 100 or len(vertices) == len(row_numbers):
                    print(
                        f"Saved {len(vertices):,}/{len(row_numbers):,} vertex lists "
                        f"({transferred / 1e6:,.0f} MB read this time).",
                        flush=True,
                    )
                    last_update = len(vertices)
        if len(vertices) != len(row_numbers):
            raise ValueError("The vertex download did not return every selected polytope.")
        return [vertices[number] for number in row_numbers]


def sample_full_catalog(h11, limit, seed, *, cache_dir=None):
    """Draw without replacement over all favorable N-lattice rows at this h11."""
    files = _parquet_files()
    offsets = _catalog_index(h11, cache_dir, files)
    ranks = random.Random(seed).sample(range(len(offsets)), min(len(offsets), limit))
    row_numbers = offsets[ranks].tolist() if ranks else []
    print(f"Selected {len(row_numbers):,} of {len(offsets):,} favorable polytopes.", flush=True)
    label = f"full KS Cornell N-favorable catalog at revision {REVISION}"
    return _selected_vertices(files, row_numbers, h11, cache_dir) if row_numbers else [], label
