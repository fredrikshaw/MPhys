"""Read sampled rows directly from the pinned catalog's Parquet shards."""

from bisect import bisect_left
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from operator import index

import pyarrow as pa
import pyarrow.parquet as pq

from MassDistParquetHttp import RangeReader


@dataclass(frozen=True)
class _RowGroup:
    url: str
    size: int
    metadata: pq.FileMetaData
    index: int
    global_start: int
    rows: tuple[int, ...]
    compressed_bytes: int


def _read_metadata(file, reader_factory):
    url, size = file
    with reader_factory(url, size) as reader:
        source = pq.ParquetFile(reader, pre_buffer=False)
        return source.metadata, reader.bytes_read


def _column_bytes(metadata, columns):
    return sum(
        column.total_compressed_size
        for column in (metadata.column(i) for i in range(metadata.num_columns))
        if column.path_in_schema.split(".", 1)[0] in columns
    )


def _plan(files, row_numbers, total_rows, workers, columns, reader_factory):
    if workers < 1:
        raise ValueError("Use at least one Parquet download worker.")
    offsets = sorted(index(row) for row in row_numbers)
    if len(set(offsets)) != len(offsets):
        raise ValueError("The sample contains a duplicate Parquet row offset.")
    if offsets and (offsets[0] < 0 or offsets[-1] >= total_rows):
        raise ValueError("A sampled Parquet row lies outside the pinned catalog.")

    planned = []
    first = offset = footer_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        metadata_results = pool.map(lambda file: _read_metadata(file, reader_factory), files)
        for (url, size), (metadata, downloaded) in zip(files, metadata_results, strict=True):
            footer_bytes += downloaded
            schema_names = set(metadata.schema.to_arrow_schema().names)
            if not set(columns) <= schema_names:
                raise ValueError(f"A required catalog column is missing from {url}.")
            for group_index in range(metadata.num_row_groups):
                group = metadata.row_group(group_index)
                stop = offset + group.num_rows
                last = bisect_left(offsets, stop, lo=first)
                if first < last:
                    planned.append(_RowGroup(
                        url, size, metadata, group_index, offset,
                        tuple(row - offset for row in offsets[first:last]),
                        _column_bytes(group, columns),
                    ))
                first = last
                offset = stop
    if offset != total_rows or first != len(offsets):
        raise ValueError("The Parquet files do not map to the expected pinned catalog size.")
    return planned, footer_bytes


def _read_group(task, h11, columns, reader_factory):
    result = []
    first = offset = 0
    # Buffered reads can stop early for a sample near the start of a group.
    # Use one read per column when the sample already reaches most of the group.
    group_rows = task.metadata.row_group(task.index).num_rows
    buffer_size = 4 * 1024 * 1024 if task.rows[-1] < group_rows * 0.8 else 0
    with reader_factory(task.url, task.size) as reader:
        source = pq.ParquetFile(
            reader, metadata=task.metadata, pre_buffer=False, buffer_size=buffer_size,
        )
        for batch in source.iter_batches(
            batch_size=65536, row_groups=[task.index], columns=columns, use_threads=False,
        ):
            stop = offset + batch.num_rows
            last = bisect_left(task.rows, stop, lo=first)
            if first < last:
                picked = batch.take(pa.array([row - offset for row in task.rows[first:last]], type=pa.int64()))
                for row, entry in zip(task.rows[first:last], picked.to_pylist(), strict=True):
                    vertices = entry["vertices"]
                    global_row = task.global_start + row
                    if not vertices or ("h11" in entry and (
                        entry["h11"] != h11 or entry["fav_N"] is not True
                    )):
                        raise ValueError(f"The sampled Parquet row {global_row} could not be verified.")
                    result.append((global_row, vertices))
            first = last
            if first == len(task.rows):
                return result, reader.bytes_read
            offset = stop
    raise ValueError("The pinned Parquet row group ended before the selected rows were read.")


def iter_parquet_vertices(
    files, row_numbers, h11, *, total_rows, workers=6, verify=True, reader_factory=RangeReader,
):
    """Yield ([(global offset, vertices), ...], transferred bytes) per row group.

    Files are (pinned URL, size) pairs in the same order used for the index.
    Results are yielded in download completion order.
    """
    files = tuple(files)
    row_numbers = tuple(row_numbers)
    if not row_numbers:
        return
    columns = ("vertices", "h11", "fav_N") if verify else ("vertices",)
    planned, footer_bytes = _plan(files, row_numbers, total_rows, workers, columns, reader_factory)
    estimated = footer_bytes + sum(task.compressed_bytes for task in planned)
    print(
        f"Retrieving {len(row_numbers):,} records across {len(planned):,} Parquet row groups "
        f"(up to {estimated / 1e6:,.0f} MB of catalog columns).",
        flush=True,
    )

    remaining = iter(planned)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def submit(task):
            return pool.submit(_read_group, task, h11, columns, reader_factory)

        pending = {submit(task) for _, task in zip(range(workers), remaining)}
        try:
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    result, transferred = future.result()
                    yield result, transferred + footer_bytes
                    footer_bytes = 0
                    task = next(remaining, None)
                    if task is not None:
                        pending.add(submit(task))
        finally:
            for future in pending:
                future.cancel()
