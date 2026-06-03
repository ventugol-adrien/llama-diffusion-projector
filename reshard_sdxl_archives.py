#!/usr/bin/env python3
"""Offline SDXL target archive re-sharder.

This script rewrites an existing SDXL target archive set (npz files containing
indices, prompt_embeds, pooled_prompt_embeds) into larger shard sizes.

Design goals:
- Keep the source archive set untouched.
- Write into a sibling destination directory.
- Use bounded writer concurrency to avoid unbounded RAM growth.
- Validate index continuity and tensor shape consistency before/after.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.lib import format as np_format

ARCHIVE_RE = re.compile(r"^target_archive_(\d{7})_(\d{7})\.npz$")


@dataclass(frozen=True)
class SourceArchive:
    path: Path
    first: int
    last: int


@dataclass
class Buffer:
    indices: np.ndarray
    prompt: np.ndarray
    pooled: np.ndarray
    rows: int = 0


def _list_archives(src_dir: Path) -> list[SourceArchive]:
    archives: list[SourceArchive] = []
    for child in src_dir.iterdir():
        if not child.is_file():
            continue
        match = ARCHIVE_RE.match(child.name)
        if not match:
            continue
        first = int(match.group(1))
        last = int(match.group(2))
        archives.append(SourceArchive(path=child, first=first, last=last))
    archives.sort(key=lambda item: item.first)
    if not archives:
        raise FileNotFoundError(f"No target_archive_*.npz files found in {src_dir}")
    return archives


def _alloc_buffer(
    rows: int, prompt_shape_tail: tuple[int, ...], pooled_dim: int, dtype: np.dtype
) -> Buffer:
    return Buffer(
        indices=np.empty((rows,), dtype=np.int64),
        prompt=np.empty((rows, *prompt_shape_tail), dtype=dtype),
        pooled=np.empty((rows, pooled_dim), dtype=dtype),
        rows=0,
    )


def _archive_filename(first: int, last: int) -> str:
    return f"target_archive_{first:07d}_{last:07d}.npz"


def _write_npz(path: Path, payload: dict[str, np.ndarray], compress: bool) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        if compress:
            np.savez_compressed(f, **payload)
        else:
            np.savez(f, **payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_archive(
    path: Path, payload: dict[str, np.ndarray], compress: bool
) -> tuple[str, int]:
    _write_npz(path, payload, compress)
    return str(path), int(payload["indices"].shape[0])


def _drain_writes(pending: list[Future], *, block: bool) -> list[Future]:
    if not pending:
        return pending
    next_pending: list[Future] = []
    for fut in pending:
        if block or fut.done():
            archive_path, rows = fut.result()
            print(f"Wrote {rows:,} rows -> {archive_path}")
        else:
            next_pending.append(fut)
    return next_pending


def _yield_source_arrays(archives: Iterable[SourceArchive]):
    for arc in archives:
        with np.load(arc.path, allow_pickle=False) as data:
            indices = data["indices"].astype(np.int64, copy=False)
            prompt = np.asarray(data["prompt_embeds"])
            pooled = np.asarray(data["pooled_prompt_embeds"])
        yield arc, indices, prompt, pooled


def _npz_member_meta(
    npz_path: Path, member_name: str
) -> tuple[tuple[int, ...], np.dtype]:
    with zipfile.ZipFile(npz_path, "r") as zf:
        member = f"{member_name}.npy"
        if member not in zf.namelist():
            raise RuntimeError(f"Missing member {member} in {npz_path.name}")
        with zf.open(member, "r") as fh:
            version = np_format.read_magic(fh)
            if version == (1, 0):
                shape, fortran_order, dtype = np_format.read_array_header_1_0(fh)
            elif version == (2, 0):
                shape, fortran_order, dtype = np_format.read_array_header_2_0(fh)
            else:
                raise RuntimeError(
                    f"Unsupported npy header version {version} in {npz_path.name}:{member}"
                )
            if fortran_order:
                raise RuntimeError(
                    f"Unexpected Fortran-order array in {npz_path.name}:{member}"
                )
            return tuple(shape), np.dtype(dtype)


def _validate_output(out_dir: Path, expected_rows: int) -> dict[str, object]:
    archives = _list_archives(out_dir)
    total_rows = 0
    first_idx = None
    last_idx = None
    missing_rows = 0
    prompt_shape = None
    pooled_shape = None
    prompt_dtype = None
    pooled_dtype = None

    for i, arc in enumerate(archives):
        if i > 0 and arc.first <= archives[i - 1].last:
            raise RuntimeError(
                f"Overlapping or unsorted ranges between {archives[i - 1].path.name} and {arc.path.name}"
            )
        if i > 0:
            missing_rows += max(0, arc.first - archives[i - 1].last - 1)

        with np.load(arc.path, allow_pickle=False) as data:
            idx = data["indices"].astype(np.int64, copy=False)

        prompt_full_shape, prompt_member_dtype = _npz_member_meta(
            arc.path, "prompt_embeds"
        )
        pooled_full_shape, pooled_member_dtype = _npz_member_meta(
            arc.path, "pooled_prompt_embeds"
        )

        if idx.ndim != 1:
            raise RuntimeError(f"indices must be 1D in {arc.path.name}")
        if prompt_full_shape[0] != idx.shape[0] or pooled_full_shape[0] != idx.shape[0]:
            raise RuntimeError(f"Row mismatch in {arc.path.name}")
        if idx.shape[0] == 0:
            raise RuntimeError(f"Empty archive {arc.path.name}")

        diffs = np.diff(idx)
        if diffs.size and not np.all(diffs > 0):
            raise RuntimeError(f"Non-increasing indices inside {arc.path.name}")
        if int(idx[0]) != arc.first or int(idx[-1]) != arc.last:
            raise RuntimeError(
                f"Filename range mismatch in {arc.path.name}: got {idx[0]}..{idx[-1]}"
            )

        if prompt_shape is None:
            prompt_shape = tuple(prompt_full_shape[1:])
            pooled_shape = tuple(pooled_full_shape[1:])
            prompt_dtype = prompt_member_dtype
            pooled_dtype = pooled_member_dtype
        else:
            if tuple(prompt_full_shape[1:]) != prompt_shape:
                raise RuntimeError(f"Prompt shape mismatch in {arc.path.name}")
            if tuple(pooled_full_shape[1:]) != pooled_shape:
                raise RuntimeError(f"Pooled shape mismatch in {arc.path.name}")
            if prompt_member_dtype != prompt_dtype:
                raise RuntimeError(f"Prompt dtype mismatch in {arc.path.name}")
            if pooled_member_dtype != pooled_dtype:
                raise RuntimeError(f"Pooled dtype mismatch in {arc.path.name}")

        total_rows += int(idx.shape[0])
        if first_idx is None:
            first_idx = int(idx[0])
        last_idx = int(idx[-1])

    if total_rows != expected_rows:
        raise RuntimeError(
            f"Row count mismatch: expected {expected_rows:,}, got {total_rows:,}"
        )

    return {
        "archive_count": len(archives),
        "rows": total_rows,
        "first_index": first_idx,
        "last_index": last_idx,
        "missing_indices_between_archives": missing_rows,
        "prompt_shape": prompt_shape,
        "pooled_shape": pooled_shape,
        "prompt_dtype": str(prompt_dtype) if prompt_dtype is not None else None,
        "pooled_dtype": str(pooled_dtype) if pooled_dtype is not None else None,
    }


def _load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(path: Path, payload: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def reshard(
    source_dir: Path,
    output_dir: Path,
    shard_rows: int,
    writer_threads: int,
    max_inflight: int,
    compress: bool,
    source_manifest: Path | None,
) -> None:
    archives = _list_archives(source_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pending: list[Future] = []
    executor = ThreadPoolExecutor(max_workers=max(1, writer_threads))

    try:
        src_iter = _yield_source_arrays(archives)
        first_arc, first_idx, first_prompt, first_pooled = next(src_iter)

        prompt_tail = tuple(first_prompt.shape[1:])
        pooled_dim = int(first_pooled.shape[1])
        target_dtype = first_prompt.dtype

        expected_rows = 0
        expected_start = int(first_idx[0])
        expected_last = int(first_idx[-1])
        expected_rows += int(first_idx.shape[0])

        buf = _alloc_buffer(shard_rows, prompt_tail, pooled_dim, target_dtype)

        def submit_payload(payload: dict[str, np.ndarray]) -> None:
            nonlocal pending
            first = int(payload["indices"][0])
            last = int(payload["indices"][-1])
            out_name = _archive_filename(first, last)
            out_path = output_dir / out_name
            while len(pending) >= max_inflight:
                pending = _drain_writes(pending, block=True)
            pending.append(executor.submit(_write_archive, out_path, payload, compress))

        def ingest_chunk(
            indices: np.ndarray, prompt: np.ndarray, pooled: np.ndarray
        ) -> None:
            nonlocal buf
            start = 0
            chunk_rows = int(indices.shape[0])
            while start < chunk_rows:
                free = shard_rows - buf.rows
                take = min(free, chunk_rows - start)
                end = start + take
                out_start = buf.rows
                out_end = out_start + take

                buf.indices[out_start:out_end] = indices[start:end]
                buf.prompt[out_start:out_end] = prompt[start:end]
                buf.pooled[out_start:out_end] = pooled[start:end]
                buf.rows = out_end

                if buf.rows == shard_rows:
                    payload = {
                        "indices": buf.indices,
                        "prompt_embeds": buf.prompt,
                        "pooled_prompt_embeds": buf.pooled,
                    }
                    submit_payload(payload)
                    buf = _alloc_buffer(
                        shard_rows, prompt_tail, pooled_dim, target_dtype
                    )

                start = end

        ingest_chunk(first_idx, first_prompt, first_pooled)

        for arc, indices, prompt, pooled in src_iter:
            if int(indices[0]) <= expected_last:
                raise RuntimeError(
                    f"Overlapping or unsorted source indices near {arc.path.name}"
                )
            expected_last = int(indices[-1])
            expected_rows += int(indices.shape[0])

            if prompt.dtype != target_dtype or pooled.dtype != target_dtype:
                raise RuntimeError(
                    f"Dtype mismatch in source {arc.path.name}: {prompt.dtype}/{pooled.dtype} != {target_dtype}"
                )
            if tuple(prompt.shape[1:]) != prompt_tail:
                raise RuntimeError(f"Prompt shape mismatch in source {arc.path.name}")
            if int(pooled.shape[1]) != pooled_dim:
                raise RuntimeError(f"Pooled shape mismatch in source {arc.path.name}")
            src_diffs = np.diff(indices)
            if src_diffs.size and not np.all(src_diffs > 0):
                raise RuntimeError(f"Non-increasing source indices in {arc.path.name}")

            ingest_chunk(indices, prompt, pooled)
            pending = _drain_writes(pending, block=False)

        if expected_start != 0:
            print(f"Warning: source starts at {expected_start}, not 0")

        if buf.rows > 0:
            payload = {
                "indices": buf.indices[: buf.rows].copy(),
                "prompt_embeds": buf.prompt[: buf.rows].copy(),
                "pooled_prompt_embeds": buf.pooled[: buf.rows].copy(),
            }
            submit_payload(payload)

        pending = _drain_writes(pending, block=True)

        summary = _validate_output(output_dir, expected_rows)
        print("Validation summary:")
        print(json.dumps(summary, indent=2, sort_keys=True))

        if source_manifest is not None and source_manifest.exists():
            manifest = _load_manifest(source_manifest)
            manifest["shard_size"] = int(shard_rows)
            out_manifest = output_dir.parent / f"manifest_{output_dir.name}.json"
            _write_manifest(out_manifest, manifest)
            print(f"Wrote destination manifest snapshot: {out_manifest}")

    finally:
        executor.shutdown(wait=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-shard SDXL target archives")
    parser.add_argument(
        "--source-dir",
        default="embedded_chunks/run-1m-2560/targets/sdxl/archive",
        help="Source SDXL target archive directory",
    )
    parser.add_argument(
        "--output-dir",
        default="embedded_chunks/run-1m-2560/targets/sdxl/archive-65536-tmp",
        help="Destination directory for re-sharded archives (must be empty or absent)",
    )
    parser.add_argument(
        "--shard-rows",
        type=int,
        default=65536,
        help="Destination rows per archive",
    )
    parser.add_argument(
        "--writer-threads",
        type=int,
        default=1,
        help="Number of writer threads",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=1,
        help="Maximum in-flight destination archives",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use np.savez_compressed for destination archives (slower)",
    )
    parser.add_argument(
        "--source-manifest",
        default="embedded_chunks/run-1m-2560/targets/sdxl/manifest.json",
        help="Source manifest path used to write a destination manifest snapshot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reshard(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        shard_rows=int(args.shard_rows),
        writer_threads=int(args.writer_threads),
        max_inflight=max(1, int(args.max_inflight)),
        compress=bool(args.compress),
        source_manifest=Path(args.source_manifest) if args.source_manifest else None,
    )


if __name__ == "__main__":
    main()
