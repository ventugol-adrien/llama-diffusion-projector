import asyncio
import aiohttp
import json
import re
import signal
import numpy as np
import pandas as pd
import os
from tqdm.asyncio import tqdm


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


# --- CONFIGURATION ---
# IMPORTANT: Point this to the batch endpoint (/v1/embeddings) of your proxy!
PROXY_URL = _env_str(
    "QWEN_PROXY_URL",
    _env_str("PROXY_URL", "https://gpu.adriens-apis.io/llm/v1/embeddings"),
)
EMBEDDING_MODEL = _env_str("QWEN_EMBEDDING_MODEL", "qwen3.5")
DATASET_PATH = _env_str(
    "QWEN_DATASET_PATH",
    _env_str("DATASET_PATH", "longclip_training_prompts.parquet"),
)
OUTPUT_DIR = _env_str(
    "QWEN_OUTPUT_DIR",
    _env_str("SOURCE_OUTPUT_DIR", "embedded_chunks"),
)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint_latest.parquet")
CHECKPOINT_TMP = CHECKPOINT_PATH + ".tmp"
ERRORS_PATH = os.path.join(OUTPUT_DIR, "errors.parquet")
ERRORS_TMP = ERRORS_PATH + ".tmp"
SOURCE_MANIFEST_PATH = os.path.join(OUTPUT_DIR, "source_manifest.json")
SOURCE_MANIFEST_TMP = SOURCE_MANIFEST_PATH + ".tmp"
LOCK_FILE = os.path.join(OUTPUT_DIR, "runner.lock")
# Two-tier checkpoint: immutable compressed archives + small rolling delta.
# Archives are written once per ARCHIVE_EVERY rows and never re-written.
# The delta covers only rows since the last archive, so writes stay fast.
ARCHIVE_DIR = os.path.join(OUTPUT_DIR, "archive")
ARCHIVE_EVERY = max(
    1,
    int(os.getenv("QWEN_ARCHIVE_EVERY", os.getenv("ARCHIVE_EVERY", "100000"))),
)  # compact once per this many rows
CHECKPOINT_DELTA = os.path.join(OUTPUT_DIR, "checkpoint_delta.parquet")
CHECKPOINT_DELTA_TMP = CHECKPOINT_DELTA + ".tmp"
BATCH_SIZE = max(1, int(os.getenv("QWEN_BATCH_SIZE", "128")))
CHECKPOINT_EVERY = max(
    1,
    int(os.getenv("QWEN_CHECKPOINT_EVERY", "1024")),
)  # save delta every N rows
DEFAULT_PROMPT_COLUMN = "prompt"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _normalize_qwen_embedding_row(row) -> np.ndarray:
    embedding = np.asarray(row, dtype=np.float32)
    if embedding.ndim not in {1, 2}:
        raise ValueError(
            f"Unsupported Qwen embedding rank {embedding.ndim}; expected 1D or 2D."
        )
    if embedding.shape[-1] <= 0:
        raise ValueError("Qwen embeddings must have a positive hidden dimension.")
    if embedding.ndim == 2 and embedding.shape[0] <= 0:
        raise ValueError(
            "Token-sequence Qwen embeddings must contain at least one token."
        )
    return embedding


def _detect_qwen_embedding_format(rows) -> str | None:
    for row in rows:
        if row is None:
            continue
        return (
            "token_sequence"
            if _normalize_qwen_embedding_row(row).ndim == 2
            else "vector"
        )
    return None


def _build_qwen_archive_payload(rows) -> tuple[dict[str, np.ndarray], str]:
    normalized_rows = [_normalize_qwen_embedding_row(row) for row in rows]
    if not normalized_rows:
        raise ValueError("Cannot build an archive payload from an empty Qwen slice.")

    row_ndims = {row.ndim for row in normalized_rows}
    if len(row_ndims) != 1:
        raise ValueError(
            "Mixed vector and token-sequence Qwen embeddings in one archive slice."
        )

    if normalized_rows[0].ndim == 1:
        return {
            "embeddings": np.stack(normalized_rows).astype(np.float32, copy=False)
        }, "vector"

    hidden_dims = {int(row.shape[1]) for row in normalized_rows}
    if len(hidden_dims) != 1:
        raise ValueError(
            "Token-sequence Qwen embeddings in one archive slice disagree on hidden size."
        )

    sequence_lengths = np.asarray(
        [row.shape[0] for row in normalized_rows], dtype=np.int32
    )
    token_offsets = np.empty(len(normalized_rows) + 1, dtype=np.int64)
    token_offsets[0] = 0
    np.cumsum(sequence_lengths, out=token_offsets[1:])
    token_embeddings = np.concatenate(normalized_rows, axis=0).astype(
        np.float32, copy=False
    )
    pooled_embeddings = np.stack(
        [row.mean(axis=0, dtype=np.float32) for row in normalized_rows]
    ).astype(np.float32, copy=False)
    return {
        "token_embeddings": token_embeddings,
        "token_offsets": token_offsets,
        "sequence_lengths": sequence_lengths,
        "pooled_embeddings": pooled_embeddings,
    }, "token_sequence"


def _resolve_prompt_column(df: pd.DataFrame) -> str:
    prompt_column = os.getenv("PROMPT_COLUMN", DEFAULT_PROMPT_COLUMN).strip()
    if not prompt_column:
        prompt_column = DEFAULT_PROMPT_COLUMN
    if prompt_column not in df.columns:
        available = ", ".join(sorted(df.columns))
        raise KeyError(
            f"Prompt column '{prompt_column}' not found in {DATASET_PATH}. "
            f"Available columns: {available}"
        )
    missing_values = int(df[prompt_column].isna().sum())
    if missing_values > 0:
        raise ValueError(
            f"Prompt column '{prompt_column}' contains {missing_values:,} missing values."
        )
    return prompt_column


def _write_source_manifest(
    prompt_column: str, embedding_format: str = "unknown"
) -> None:
    payload = {
        "schema_version": 1,
        "dataset_path": DATASET_PATH,
        "prompt_column": prompt_column,
        "embedding_format": embedding_format,
        "embedding_model": EMBEDDING_MODEL,
        "proxy_url": PROXY_URL,
        "output_dir": OUTPUT_DIR,
        "archive_every": ARCHIVE_EVERY,
    }
    with open(SOURCE_MANIFEST_TMP, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(SOURCE_MANIFEST_TMP, SOURCE_MANIFEST_PATH)


def acquire_lock():
    """Exit if another instance is already running; write our PID otherwise."""
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE) as f:
            pid_str = f.read().strip()
        try:
            pid = int(pid_str)
            os.kill(pid, 0)  # signal 0: just probe whether the process exists
            print(f"\nError: runner.py is already running (PID {pid}).")
            print(f"To force a restart, delete {LOCK_FILE} and try again.")
            raise SystemExit(1)
        except ProcessLookupError:
            print(f"Removing stale lock (PID {pid_str} is no longer running).")
        except ValueError:
            pass  # corrupt lock file — overwrite it
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))


def release_lock():
    try:
        os.remove(LOCK_FILE)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Two-tier checkpoint helpers
# ---------------------------------------------------------------------------


def _archive_filename(first_pos: int, last_pos: int) -> str:
    """Canonical archive path for the row-position range [first_pos, last_pos]."""
    return os.path.join(ARCHIVE_DIR, f"archive_{first_pos:07d}_{last_pos:07d}.npz")


def _list_archives() -> list[str]:
    """Sorted list of existing archive file paths."""
    if not os.path.exists(ARCHIVE_DIR):
        return []
    return sorted(
        os.path.join(ARCHIVE_DIR, f)
        for f in os.listdir(ARCHIVE_DIR)
        if f.startswith("archive_") and f.endswith(".npz")
    )


def get_archive_max_pos() -> int:
    """Highest row position covered by any archive, or -1 if none exist."""
    archives = _list_archives()
    if not archives:
        return -1
    # Filename: archive_FFFFFFF_LLLLLLL.npz  →  last pos = LLLLLLL
    return int(os.path.basename(archives[-1]).split("_")[2].split(".")[0])


def write_archive(df: pd.DataFrame, first_pos: int, rows_done: int) -> int:
    """
    Atomically compress rows [first_pos, rows_done) into a .npz archive.
    Clears the delta checkpoint afterwards (it is now superseded).
    Returns the last row position covered (= rows_done - 1).
    """
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    slice_df = df.iloc[first_pos:rows_done][["qwen_embedding"]].dropna()
    if len(slice_df) > 0:
        indices = np.array(slice_df.index, dtype=np.int64)
        payload, embedding_format = _build_qwen_archive_payload(
            slice_df["qwen_embedding"].tolist()
        )
        fname = _archive_filename(first_pos, rows_done - 1)
        tmp_base = os.path.join(ARCHIVE_DIR, "_tmp_archive")  # numpy appends .npz
        np.savez_compressed(tmp_base, indices=indices, **payload)
        with open(tmp_base + ".npz", "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_base + ".npz", fname)
        print(
            f"\nArchived {len(slice_df):,} embeddings "
            f"(pos {first_pos:,}–{rows_done - 1:,}, format={embedding_format}) → {os.path.basename(fname)}"
        )
    # Clear delta: it is fully covered by the new archive
    for p in [CHECKPOINT_DELTA, CHECKPOINT_DELTA_TMP]:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
    return rows_done - 1


def save_checkpoint_delta(df: pd.DataFrame, archive_max_pos: int, rows_done: int):
    """
    Atomically save only rows newer than archive_max_pos to the delta checkpoint.
    The delta is at most ARCHIVE_EVERY rows, so writes stay fast and small.
    """
    start = archive_max_pos + 1
    if start >= rows_done:
        return
    ckpt_df = df.iloc[start:rows_done][["qwen_embedding"]].dropna()
    if len(ckpt_df) == 0:
        return
    ckpt_df.to_parquet(CHECKPOINT_DELTA_TMP)
    with open(CHECKPOINT_DELTA_TMP, "rb") as f:
        os.fsync(f.fileno())
    os.replace(CHECKPOINT_DELTA_TMP, CHECKPOINT_DELTA)


def save_errors(failed_rows):
    """
    Atomically merge new failures into errors.parquet.

    Loads any existing errors file, appends new failures, deduplicates by
    original_index, then writes atomically (tmp + fsync + replace).
    """
    if not failed_rows:
        return

    new_df = pd.DataFrame(
        [
            {"original_index": idx, "prompt": prompt}
            for idx, prompt in failed_rows.items()
        ]
    )

    if os.path.exists(ERRORS_PATH):
        try:
            existing = pd.read_parquet(ERRORS_PATH)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["original_index"], keep="last")
        except Exception:
            combined = new_df
    else:
        combined = new_df

    combined.to_parquet(ERRORS_TMP)
    with open(ERRORS_TMP, "rb") as f:
        os.fsync(f.fileno())
    os.replace(ERRORS_TMP, ERRORS_PATH)
    print(f"\nErrors file updated: {len(combined):,} total failed rows → {ERRORS_PATH}")


def _parse_bad_item_index(error_text: str) -> int | None:
    """Extract the item index from a proxy error like: {"detail":"llama.cpp returned 400 for item 10"}"""
    try:
        detail = json.loads(error_text).get("detail", "")
    except Exception:
        detail = error_text
    m = re.search(r"\bitem\s+(\d+)", detail, re.IGNORECASE)
    return int(m.group(1)) if m else None


async def fetch_batch_embeddings(session, batch_prompts):
    """
    Fetch embeddings for a batch.  If the proxy identifies a specific bad item
    (e.g. {"detail": "llama.cpp returned 400 for item 10"}), that item is
    stripped and the remaining prompts are retried.  The loop continues until
    the sub-batch succeeds or an unidentifiable error halts it.

    Returns a list parallel to batch_prompts; failed items are None.
    """
    timeout = aiohttp.ClientTimeout(total=300.0)

    # pending: list of (original_offset, prompt) so we can map results back
    pending = list(enumerate(batch_prompts))
    results = [None] * len(batch_prompts)

    while pending:
        prompts = [p for _, p in pending]
        payload = {"input": prompts, "model": EMBEDDING_MODEL}

        try:
            async with session.post(
                PROXY_URL, json=payload, timeout=timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    embeddings = [item["embedding"] for item in data.get("data", [])]
                    for (orig_offset, _), emb in zip(pending, embeddings):
                        results[orig_offset] = emb
                    break  # entire pending set succeeded

                else:
                    error_text = await response.text()
                    print(f"\nProxy Error {response.status}: {error_text}")

                    bad_idx = _parse_bad_item_index(error_text)
                    if bad_idx is not None and bad_idx < len(pending):
                        orig_offset, bad_prompt = pending[bad_idx]
                        print(
                            f"  → Dropping item {orig_offset} from batch "
                            f"(sub-batch idx {bad_idx}), retrying the rest."
                        )
                        pending.pop(bad_idx)  # remove just the bad item and retry
                    else:
                        # Can't identify which item failed — fail the whole remaining batch
                        break

        except Exception as e:
            print(f"\nBatch request failed: {e}")
            break

    return results


async def process_dataset():
    # Use asyncio's signal integration so SIGTERM raises CancelledError at the
    # next await point — the try/finally block runs cleanly with no traceback.
    if hasattr(signal, "SIGTERM"):  # not available on Windows
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)

    acquire_lock()
    print("Loading dataset...")
    print(f"Qwen output root: {OUTPUT_DIR}")
    print(f"Qwen archive dir: {ARCHIVE_DIR}")
    print(f"Qwen archive rows: {ARCHIVE_EVERY:,}")
    print(f"Qwen proxy URL: {PROXY_URL}")
    df = pd.read_parquet(DATASET_PATH)
    prompt_column = _resolve_prompt_column(df)
    total_rows = df.shape[0]
    df["qwen_embedding"] = None
    print(f"Using prompt column: {prompt_column}")
    recorded_embedding_format = "unknown"
    _write_source_manifest(prompt_column, recorded_embedding_format)

    # ----------------------------------------------------------------
    # One-time migration: convert legacy checkpoint_latest.parquet → archives
    # ----------------------------------------------------------------
    if (
        os.path.exists(CHECKPOINT_PATH)
        and not _list_archives()
        and not os.path.exists(CHECKPOINT_DELTA)
    ):
        print(f"Migrating {CHECKPOINT_PATH} to archive format (one-time)...")
        try:
            old_ckpt = pd.read_parquet(CHECKPOINT_PATH)
            old_ckpt["qwen_embedding"] = old_ckpt["qwen_embedding"].apply(
                lambda x: x.tolist() if hasattr(x, "tolist") else x
            )
            os.makedirs(ARCHIVE_DIR, exist_ok=True)
            max_old_pos = int(old_ckpt.index.max())
            for chunk_start in range(0, max_old_pos + 1, ARCHIVE_EVERY):
                chunk_end = min(chunk_start + ARCHIVE_EVERY - 1, max_old_pos)
                chunk = old_ckpt[
                    (old_ckpt.index >= chunk_start) & (old_ckpt.index <= chunk_end)
                ]
                indices = np.array(chunk.index, dtype=np.int64)
                embeddings = np.array(
                    chunk["qwen_embedding"].tolist(), dtype=np.float32
                )
                tmp_base = os.path.join(ARCHIVE_DIR, "_tmp_archive")
                np.savez_compressed(tmp_base, indices=indices, embeddings=embeddings)
                with open(tmp_base + ".npz", "rb") as f:
                    os.fsync(f.fileno())
                os.replace(tmp_base + ".npz", _archive_filename(chunk_start, chunk_end))
                print(f"  Archived rows {chunk_start:,}–{chunk_end:,}")
            del old_ckpt
            os.rename(CHECKPOINT_PATH, CHECKPOINT_PATH + ".migrated")
            print("Migration complete. Old file kept as .migrated for safety.")
        except Exception as e:
            print(f"Migration failed ({e}). Falling back to old checkpoint.")

    # Discard any stale delta temp file left by a previous crash
    if os.path.exists(CHECKPOINT_DELTA_TMP):
        print("Removing stale delta .tmp left by a previous crash.")
        os.remove(CHECKPOINT_DELTA_TMP)

    # Archives tell us the last committed position; delta has the rest
    archive_max_pos = get_archive_max_pos()
    delta_max_pos = -1
    if os.path.exists(CHECKPOINT_DELTA):
        print(f"Loading delta checkpoint: {CHECKPOINT_DELTA}")
        try:
            delta = pd.read_parquet(CHECKPOINT_DELTA)
            delta["qwen_embedding"] = delta["qwen_embedding"].apply(
                lambda x: x.tolist() if hasattr(x, "tolist") else x
            )
            df.loc[delta.index, "qwen_embedding"] = delta["qwen_embedding"]
            delta_max_pos = int(delta.index.max())
            del delta
        except Exception as e:
            print(f"Delta checkpoint corrupted ({e}), discarding.")
            os.remove(CHECKPOINT_DELTA)

    last_saved_pos = max(archive_max_pos, delta_max_pos)
    if last_saved_pos >= 0:
        start_i = ((last_saved_pos + 1) // BATCH_SIZE) * BATCH_SIZE
    else:
        start_i = 0

    if start_i >= total_rows:
        print("All rows already embedded!")
        return

    if last_saved_pos >= 0:
        print(
            f"Resuming from row {start_i:,} / {total_rows:,} "
            f"(archive up to {archive_max_pos:,}, delta up to {delta_max_pos:,})"
        )
    print(f"Processing {total_rows - start_i:,} remaining prompts...")

    failed_rows: dict[int, str] = {}  # {original_df_index: prompt}

    # Hard per-request timeout: if the backend stalls (not 503, but actually hangs),
    # each request will fail after 120 s instead of blocking the runner forever.
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        pbar = tqdm(total=total_rows - start_i, desc="Processing Prompts")
        i = start_i
        try:
            for i in range(start_i, total_rows, BATCH_SIZE):
                batch_slice = df[prompt_column].iloc[i : i + BATCH_SIZE].tolist()
                embeddings = await fetch_batch_embeddings(session, batch_slice)

                if len(embeddings) == len(batch_slice):
                    df.loc[i : i + BATCH_SIZE - 1, "qwen_embedding"] = pd.array(
                        embeddings, dtype=object
                    )
                    detected_format = _detect_qwen_embedding_format(embeddings)
                    if (
                        detected_format is not None
                        and detected_format != recorded_embedding_format
                    ):
                        recorded_embedding_format = detected_format
                        _write_source_manifest(prompt_column, recorded_embedding_format)
                    # Record any individual None embeddings within the batch
                    for offset, emb in enumerate(embeddings):
                        if emb is None:
                            failed_rows[i + offset] = batch_slice[offset]
                else:
                    # Entire batch lost (count mismatch) — mark all as failed
                    print(
                        f"Warning: Proxy returned mismatched embedding count at index {i}"
                    )
                    for offset, prompt in enumerate(batch_slice):
                        failed_rows[i + offset] = prompt

                pbar.update(len(batch_slice))

                # Periodic checkpoint
                rows_done = i + BATCH_SIZE
                if rows_done % CHECKPOINT_EVERY < BATCH_SIZE:
                    try:
                        save_checkpoint_delta(df, archive_max_pos, rows_done)
                        print(f"\nCheckpoint saved ({rows_done:,} rows)")
                    except Exception as ckpt_err:
                        print(
                            f"\nWARNING: checkpoint failed at row {rows_done:,}: {ckpt_err}"
                        )
                    # Archive when the delta has grown large enough
                    if rows_done - (archive_max_pos + 1) >= ARCHIVE_EVERY:
                        try:
                            archive_max_pos = write_archive(
                                df, archive_max_pos + 1, rows_done
                            )
                        except Exception as arc_err:
                            print(f"\nWARNING: archive failed: {arc_err}")

        finally:
            pbar.close()
            release_lock()
            save_errors(failed_rows)
            # Always save progress on any exit (Ctrl+C, error, or normal completion)
            rows_done = min(i + BATCH_SIZE, total_rows)
            if rows_done < total_rows:
                try:
                    save_checkpoint_delta(df, archive_max_pos, rows_done)
                    print(f"\nProgress saved at row {rows_done:,}. Re-run to resume.")
                except Exception as e:
                    print(f"\nWARNING: shutdown save failed: {e}")
                return

    # Write the final archive for any remaining unarchived rows
    remaining_start = archive_max_pos + 1
    if remaining_start < total_rows:
        try:
            write_archive(df, remaining_start, total_rows)
        except Exception as arc_err:
            print(f"\nWARNING: final archive failed: {arc_err}")

    print(f"\nPipeline Complete! {total_rows:,} rows processed.")
    if failed_rows:
        print(f"{len(failed_rows):,} rows failed — see {ERRORS_PATH} to retry.")
    print("Run clip_runner.py next to add CLIP embeddings.")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(process_dataset())
