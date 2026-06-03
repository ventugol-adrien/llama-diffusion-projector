import json
import os
from collections import deque
from contextlib import suppress
from time import perf_counter

# RDNA4 (gfx1200) can require an explicit override on ROCm so PyTorch builds
# the correct kernels instead of falling back to slower generic behavior.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "12.0.0")

import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from target_families import (
    build_target_manifest,
    ensure_target_root,
    get_target_family,
    get_target_family_spec,
    get_target_layout,
    write_target_manifest,
)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


# --- CONFIGURATION ---
DATASET_PATH = _env_str(
    "QWEN_DATASET_PATH",
    _env_str("DATASET_PATH", "longclip_training_prompts.parquet"),
)
SOURCE_OUTPUT_DIR = _env_str(
    "QWEN_OUTPUT_DIR",
    _env_str("SOURCE_OUTPUT_DIR", "embedded_chunks"),
)
TARGET_OUTPUT_DIR = _env_str("TARGET_OUTPUT_DIR", "embedded_chunks")
QWEN_CHECKPOINT = os.path.join(SOURCE_OUTPUT_DIR, "checkpoint_latest.parquet")
CLIP_CHECKPOINT_PATH = os.path.join(TARGET_OUTPUT_DIR, "clip_checkpoint_latest.parquet")
CLIP_CHECKPOINT_TMP = CLIP_CHECKPOINT_PATH + ".tmp"
CLIP_ERRORS_PATH = os.path.join(TARGET_OUTPUT_DIR, "clip_errors.parquet")
CLIP_ERRORS_TMP = CLIP_ERRORS_PATH + ".tmp"
SOURCE_MANIFEST_PATH = os.path.join(SOURCE_OUTPUT_DIR, "source_manifest.json")
FINAL_OUTPUT = "longclip_training_prompts_with_embeddings.parquet"
FINAL_OUTPUT_TMP = FINAL_OUTPUT + ".tmp"
# Two-tier CLIP checkpoint: immutable archives + small rolling delta.
CLIP_ARCHIVE_DIR = os.path.join(TARGET_OUTPUT_DIR, "clip_archive")
CLIP_ARCHIVE_EVERY = int(os.getenv("CLIP_ARCHIVE_EVERY", "250000"))
CLIP_CHECKPOINT_DELTA = os.path.join(TARGET_OUTPUT_DIR, "clip_checkpoint_delta.parquet")
CLIP_CHECKPOINT_DELTA_TMP = CLIP_CHECKPOINT_DELTA + ".tmp"
SDXL_CLIP_ARCHIVE_EVERY = int(os.getenv("SDXL_CLIP_ARCHIVE_EVERY", "0"))

# LongCLIP-L: extends CLIP ViT-L/14 context from 77 to 248 tokens.
# Public, no gating. Outputs 768-dim pooler_output.
CLIP_MODEL_ID = "zer0int/LongCLIP-L-Diffusers"
LONG_CLIP_MAX_LENGTH = 248
SDXL_PROMPT_COLUMN = "prompt_embeds"
SDXL_POOLED_COLUMN = "pooled_prompt_embeds"

# Two-tier Qwen checkpoint locations (mirrors runner.py)
ARCHIVE_DIR = os.path.join(SOURCE_OUTPUT_DIR, "archive")
CHECKPOINT_DELTA = os.path.join(SOURCE_OUTPUT_DIR, "checkpoint_delta.parquet")

BATCH_SIZE = int(os.getenv("CLIP_BATCH_SIZE", "512"))
CLIP_MAX_ROWS = int(os.getenv("CLIP_MAX_ROWS", "0"))
TARGET_ARCHIVE_COMPRESS = os.getenv(
    "TARGET_ARCHIVE_COMPRESS", "1"
).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
SDXL_TARGET_DTYPE = os.getenv("SDXL_TARGET_DTYPE", "float16").strip().lower()
if SDXL_TARGET_DTYPE not in {"float16", "float32"}:
    raise ValueError(
        f"Unsupported SDXL_TARGET_DTYPE '{SDXL_TARGET_DTYPE}'. Expected float16 or float32."
    )
TARGET_ARCHIVE_WRITERS = max(0, int(os.getenv("TARGET_ARCHIVE_WRITERS", "7")))
TARGET_ARCHIVE_MAX_INFLIGHT = max(
    1,
    int(
        os.getenv(
            "TARGET_ARCHIVE_MAX_INFLIGHT",
            str(max(1, TARGET_ARCHIVE_WRITERS)),
        )
    ),
)
CHECKPOINT_EVERY = int(os.getenv("CLIP_CHECKPOINT_EVERY", "100000"))
SDXL_CHECKPOINT_EVERY = int(os.getenv("SDXL_CHECKPOINT_EVERY", "50000"))
# Sequence lengths are rounded up to multiples of BUCKET_SIZE before the
# forward pass.  This limits the number of distinct tensor shapes to at most
# LONG_CLIP_MAX_LENGTH // BUCKET_SIZE, so ROCm only needs to JIT-compile
# kernels once per bucket rather than once per unique batch length.
BUCKET_SIZE = 32
# Number of batches to tokenize ahead of the inference loop.
# Tokenization is CPU-light; keeping several batches ready means the
# forward-pass thread never stalls waiting for input.
DEFAULT_PREFETCH = (
    2 if (os.cpu_count() or 4) <= 8 else max(4, (os.cpu_count() or 4) // 2)
)
PREFETCH = max(1, int(os.getenv("CLIP_PREFETCH", str(DEFAULT_PREFETCH))))
# Env-gated timing instrumentation for diagnosis. Set CLIP_TIMING_BATCHES to a
# positive number to log that many early batches, or -1 to log every batch.
TIMING_BATCHES = int(os.getenv("CLIP_TIMING_BATCHES", "0"))
TIMING_EVERY = max(1, int(os.getenv("CLIP_TIMING_EVERY", "1")))
DEFAULT_PROMPT_COLUMN = "prompt"

os.makedirs(SOURCE_OUTPUT_DIR, exist_ok=True)
os.makedirs(TARGET_OUTPUT_DIR, exist_ok=True)


def _requested_prompt_column() -> str:
    prompt_column = os.getenv("PROMPT_COLUMN", DEFAULT_PROMPT_COLUMN).strip()
    return prompt_column or DEFAULT_PROMPT_COLUMN


def _resolve_prompt_column(df: pd.DataFrame, prompt_column: str | None = None) -> str:
    prompt_column = prompt_column or _requested_prompt_column()
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


def _validate_source_manifest(prompt_column: str) -> None:
    if not os.path.exists(SOURCE_MANIFEST_PATH):
        return
    with open(SOURCE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    recorded_prompt_column = payload.get("prompt_column")
    if recorded_prompt_column and recorded_prompt_column != prompt_column:
        raise RuntimeError(
            "Prompt contract mismatch between runner.py and clip_runner.py: "
            f"source manifest recorded '{recorded_prompt_column}', "
            f"but clip_runner.py is using '{prompt_column}'."
        )

    recorded_dataset_path = payload.get("dataset_path")
    if recorded_dataset_path and recorded_dataset_path != DATASET_PATH:
        raise RuntimeError(
            "Dataset path mismatch between runner.py and clip_runner.py: "
            f"source manifest recorded '{recorded_dataset_path}', "
            f"but clip_runner.py is configured for '{DATASET_PATH}'."
        )


def _sdxl_target_numpy_dtype():
    return np.float16 if SDXL_TARGET_DTYPE == "float16" else np.float32


def _sdxl_target_torch_dtype():
    return torch.float16 if SDXL_TARGET_DTYPE == "float16" else torch.float32


def _target_archive_dtype_name(target_family: str) -> str:
    if target_family == "sdxl":
        return SDXL_TARGET_DTYPE
    return "float32"


def _log_gpu_runtime(device):
    if device.type != "cuda":
        return
    if torch.version.hip is not None:
        print(
            "ROCm launch: "
            f"HSA_OVERRIDE_GFX_VERSION={os.getenv('HSA_OVERRIDE_GFX_VERSION')}, "
            f"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL={os.getenv('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL')}"
        )
        return
    if torch.version.cuda is not None:
        print(f"CUDA runtime: torch.version.cuda={torch.version.cuda}")


# ==========================================
# ARCHIVE HELPERS (Qwen two-tier checkpoint)
# ==========================================


def _list_archives() -> list[str]:
    """Sorted list of existing Qwen archive file paths."""
    if not os.path.exists(ARCHIVE_DIR):
        return []
    return sorted(
        os.path.join(ARCHIVE_DIR, f)
        for f in os.listdir(ARCHIVE_DIR)
        if f.startswith("archive_") and f.endswith(".npz")
    )


def _sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _should_log_timing(batch_num: int) -> bool:
    if TIMING_BATCHES == 0:
        return False
    if TIMING_BATCHES > 0 and batch_num > TIMING_BATCHES:
        return False
    return batch_num % TIMING_EVERY == 0


def load_qwen_embeddings() -> pd.DataFrame | None:
    """
    Return a DataFrame whose *index* is the set of row positions that runner.py
    has already embedded.  Only indices are loaded — not the embedding vectors —
    to avoid large memory use (Qwen embeddings are no longer needed here).
    """
    archives = _list_archives()

    if archives:
        print(f"Scanning {len(archives)} Qwen archive(s) for row indices...")
        all_indices: list[np.ndarray] = []
        for archive_path in archives:
            data = np.load(archive_path, allow_pickle=False)
            all_indices.append(data["indices"].astype(np.int64))
            del data

        if os.path.exists(CHECKPOINT_DELTA):
            try:
                delta_idx = pd.read_parquet(CHECKPOINT_DELTA, columns=[]).index
                all_indices.append(delta_idx.to_numpy(dtype=np.int64))
            except Exception as e:
                print(f"Delta checkpoint unreadable ({e}), skipping.")

        combined = np.unique(np.concatenate(all_indices))
        print(f"{len(combined):,} rows with Qwen embeddings found.")
        return pd.DataFrame(index=combined)

    # Fallback: legacy checkpoint — read only the index, not the embedding column
    if os.path.exists(QWEN_CHECKPOINT):
        print(f"Reading index from Qwen checkpoint: {QWEN_CHECKPOINT}")
        return pd.read_parquet(QWEN_CHECKPOINT, columns=[])

    return None


# ==========================================
# ATOMIC I/O HELPERS
# ==========================================


def _atomic_write(df, tmp_path, final_path):
    """Write df to tmp, fsync, then atomically replace final_path."""
    df.to_parquet(tmp_path)
    with open(tmp_path, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp_path, final_path)


def save_clip_errors(failed_rows):
    """Atomically merge new failures into clip_errors.parquet."""
    if not failed_rows:
        return
    new_df = pd.DataFrame(
        [
            {"original_index": idx, "prompt": prompt}
            for idx, prompt in failed_rows.items()
        ]
    )
    if os.path.exists(CLIP_ERRORS_PATH):
        try:
            existing = pd.read_parquet(CLIP_ERRORS_PATH)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["original_index"], keep="last")
        except Exception:
            combined = new_df
    else:
        combined = new_df
    _atomic_write(combined, CLIP_ERRORS_TMP, CLIP_ERRORS_PATH)
    print(
        f"\nCLIP errors file updated: {len(combined):,} total failed rows → {CLIP_ERRORS_PATH}"
    )


def _list_target_archives(archive_dir, prefix) -> list[str]:
    if not os.path.exists(archive_dir):
        return []
    return sorted(
        os.path.join(archive_dir, f)
        for f in os.listdir(archive_dir)
        if f.startswith(prefix) and f.endswith(".npz")
    )


def _resolve_active_target_paths(target_layout):
    archives = _list_target_archives(
        target_layout.archive_dir, target_layout.archive_prefix
    )
    if archives:
        return {
            "archive_dir": target_layout.archive_dir,
            "archive_prefix": target_layout.archive_prefix,
            "checkpoint_delta_path": target_layout.checkpoint_delta_path,
            "checkpoint_delta_tmp_path": target_layout.checkpoint_delta_tmp_path,
            "errors_path": target_layout.errors_path,
            "errors_tmp_path": target_layout.errors_tmp_path,
            "archives": archives,
            "using_legacy": False,
        }

    if (
        target_layout.legacy_archive_dir is not None
        and target_layout.legacy_archive_prefix is not None
    ):
        legacy_archives = _list_target_archives(
            target_layout.legacy_archive_dir,
            target_layout.legacy_archive_prefix,
        )
        if legacy_archives:
            return {
                "archive_dir": target_layout.legacy_archive_dir,
                "archive_prefix": target_layout.legacy_archive_prefix,
                "checkpoint_delta_path": CLIP_CHECKPOINT_DELTA,
                "checkpoint_delta_tmp_path": CLIP_CHECKPOINT_DELTA_TMP,
                "errors_path": CLIP_ERRORS_PATH,
                "errors_tmp_path": CLIP_ERRORS_TMP,
                "archives": legacy_archives,
                "using_legacy": True,
            }

    return {
        "archive_dir": target_layout.archive_dir,
        "archive_prefix": target_layout.archive_prefix,
        "checkpoint_delta_path": target_layout.checkpoint_delta_path,
        "checkpoint_delta_tmp_path": target_layout.checkpoint_delta_tmp_path,
        "errors_path": target_layout.errors_path,
        "errors_tmp_path": target_layout.errors_tmp_path,
        "archives": [],
        "using_legacy": False,
    }


# ==========================================# CLIP ARCHIVE HELPERS (two-tier checkpoint)
# ==========================================


def _target_archive_filename(
    archive_dir, archive_prefix, first_pos: int, last_pos: int
):
    return os.path.join(
        archive_dir, f"{archive_prefix}{first_pos:07d}_{last_pos:07d}.npz"
    )


def get_clip_archive_max_pos(archives) -> int:
    if not archives:
        return -1
    return int(os.path.splitext(os.path.basename(archives[-1]))[0].rsplit("_", 1)[1])


def _target_columns(target_family: str) -> list[str]:
    if target_family == "sdxl":
        return [SDXL_PROMPT_COLUMN, SDXL_POOLED_COLUMN]
    return ["clip_embedding"]


def _checkpoint_every(target_family: str) -> int:
    if target_family == "sdxl":
        return SDXL_CHECKPOINT_EVERY
    return CHECKPOINT_EVERY


def _archive_every(target_family: str) -> int:
    if target_family == "sdxl":
        if SDXL_CLIP_ARCHIVE_EVERY > 0:
            return SDXL_CLIP_ARCHIVE_EVERY
        return BATCH_SIZE
    return CLIP_ARCHIVE_EVERY


def _target_input_length(inputs, target_family: str) -> int:
    if target_family == "sdxl":
        return int(inputs["primary"]["input_ids"].shape[1])
    return int(inputs["input_ids"].shape[1])


def _initialize_target_columns(df: pd.DataFrame, target_family: str):
    for column in _target_columns(target_family):
        df[column] = None


def _merge_target_rows(df: pd.DataFrame, source_df: pd.DataFrame, target_family: str):
    for column in _target_columns(target_family):
        df.loc[source_df.index, column] = source_df[column]


def _clear_target_rows(
    df: pd.DataFrame, first_pos: int, rows_done: int, target_family: str
):
    column_positions = [
        df.columns.get_loc(column) for column in _target_columns(target_family)
    ]
    df.iloc[first_pos:rows_done, column_positions] = None


def _store_target_value(
    df: pd.DataFrame,
    row_pos: int,
    encoded_value,
    target_family: str,
    column_positions: dict[str, int],
):
    if target_family == "sdxl":
        prompt_embeds, pooled_prompt_embeds = encoded_value
        df.iat[row_pos, column_positions[SDXL_PROMPT_COLUMN]] = prompt_embeds
        df.iat[row_pos, column_positions[SDXL_POOLED_COLUMN]] = pooled_prompt_embeds
        return
    df.iat[row_pos, column_positions["clip_embedding"]] = encoded_value


def _resolve_hf_source(source: str) -> tuple[str, str | None]:
    parts = source.split("/")
    if len(parts) <= 2:
        return source, None
    return "/".join(parts[:2]), "/".join(parts[2:])


def _load_hf_component(component_cls, source: str, **kwargs):
    repo_id, subfolder = _resolve_hf_source(source)
    if subfolder is not None:
        kwargs["subfolder"] = subfolder
    return component_cls.from_pretrained(repo_id, **kwargs)


def _archive_payload_from_slice(slice_df: pd.DataFrame, target_family: str) -> dict:
    payload = {"indices": np.array(slice_df.index, dtype=np.int64)}
    if target_family == "sdxl":
        payload["prompt_embeds"] = np.asarray(
            slice_df[SDXL_PROMPT_COLUMN].tolist(),
            dtype=_sdxl_target_numpy_dtype(),
        )
        payload["pooled_prompt_embeds"] = np.asarray(
            slice_df[SDXL_POOLED_COLUMN].tolist(),
            dtype=_sdxl_target_numpy_dtype(),
        )
    else:
        payload["embeddings"] = np.asarray(
            slice_df["clip_embedding"].tolist(), dtype=np.float32
        )
    return payload


def _write_npz_payload(target, **payload):
    if TARGET_ARCHIVE_COMPRESS:
        np.savez_compressed(target, **payload)
        return
    np.savez(target, **payload)


def _write_npz_file(npz_path: str, payload: dict, tmp_path: str | None = None):
    tmp_path = npz_path + ".tmp" if tmp_path is None else tmp_path
    with open(tmp_path, "wb") as f:
        _write_npz_payload(f, **payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, npz_path)


def _remove_files(*paths: str):
    for path in paths:
        with suppress(FileNotFoundError):
            os.remove(path)


def _new_sdxl_payload_buffer() -> dict:
    return {
        "positions": deque(),
        "indices": deque(),
        "prompt_embeds": deque(),
        "pooled_prompt_embeds": deque(),
        "rows": 0,
    }


def _append_sdxl_payload_buffer(
    buffer: dict,
    positions: np.ndarray,
    indices: np.ndarray,
    prompt_embeds: np.ndarray,
    pooled_prompt_embeds: np.ndarray,
):
    if len(indices) == 0:
        return
    buffer["positions"].append(positions)
    buffer["indices"].append(indices)
    buffer["prompt_embeds"].append(prompt_embeds)
    buffer["pooled_prompt_embeds"].append(pooled_prompt_embeds)
    buffer["rows"] += len(indices)


def _merge_sdxl_payloads(payloads: list[dict]) -> dict | None:
    if not payloads:
        return None
    if len(payloads) == 1:
        return payloads[0]
    return {
        "indices": np.concatenate([payload["indices"] for payload in payloads], axis=0),
        "prompt_embeds": np.concatenate(
            [payload["prompt_embeds"] for payload in payloads], axis=0
        ),
        "pooled_prompt_embeds": np.concatenate(
            [payload["pooled_prompt_embeds"] for payload in payloads], axis=0
        ),
    }


def _snapshot_sdxl_payload_buffer(buffer: dict) -> dict | None:
    if buffer["rows"] == 0:
        return None
    return {
        "indices": np.concatenate(list(buffer["indices"]), axis=0),
        "prompt_embeds": np.concatenate(list(buffer["prompt_embeds"]), axis=0),
        "pooled_prompt_embeds": np.concatenate(
            list(buffer["pooled_prompt_embeds"]), axis=0
        ),
    }


def _pop_sdxl_payload_buffer(buffer: dict, last_pos: int) -> dict | None:
    if buffer["rows"] == 0:
        return None
    payload_parts: list[dict] = []
    while buffer["positions"]:
        positions = buffer["positions"][0]
        if positions[0] > last_pos:
            break
        indices = buffer["indices"][0]
        prompt_embeds = buffer["prompt_embeds"][0]
        pooled_prompt_embeds = buffer["pooled_prompt_embeds"][0]
        take = int(np.searchsorted(positions, last_pos, side="right"))
        if take <= 0:
            break
        payload_parts.append(
            {
                "indices": indices[:take],
                "prompt_embeds": prompt_embeds[:take],
                "pooled_prompt_embeds": pooled_prompt_embeds[:take],
            }
        )
        if take == len(indices):
            buffer["positions"].popleft()
            buffer["indices"].popleft()
            buffer["prompt_embeds"].popleft()
            buffer["pooled_prompt_embeds"].popleft()
        else:
            buffer["positions"][0] = positions[take:]
            buffer["indices"][0] = indices[take:]
            buffer["prompt_embeds"][0] = prompt_embeds[take:]
            buffer["pooled_prompt_embeds"][0] = pooled_prompt_embeds[take:]
        buffer["rows"] -= take
    return _merge_sdxl_payloads(payload_parts)


def _sdxl_checkpoint_payload(
    pending_archives: deque, pending_buffer: dict
) -> dict | None:
    payloads = [meta["payload"] for meta in pending_archives]
    tail_payload = _snapshot_sdxl_payload_buffer(pending_buffer)
    if tail_payload is not None:
        payloads.append(tail_payload)
    return _merge_sdxl_payloads(payloads)


def _write_target_archive_payload(archive_path: str, payload: dict):
    _write_npz_file(archive_path, payload)


def _drain_completed_target_archives(
    pending_archives: deque,
    *,
    block: bool = False,
) -> int | None:
    if not pending_archives:
        return None
    if block and not pending_archives[0]["future"].done():
        pending_archives[0]["future"].result()

    durable_max_pos = None
    while pending_archives and pending_archives[0]["future"].done():
        meta = pending_archives.popleft()
        meta["future"].result()
        print(
            f"\nArchived {len(meta['payload']['indices']):,} SDXL target rows "
            f"(pos {meta['first_pos']:,}–{meta['last_pos']:,}) → {os.path.basename(meta['archive_path'])}"
        )
        durable_max_pos = meta["last_pos"]
    return durable_max_pos


def _save_sdxl_checkpoint_payload(
    payload: dict | None,
    checkpoint_path: str,
    checkpoint_tmp_path: str,
):
    if payload is None or len(payload["indices"]) == 0:
        return
    _write_npz_file(checkpoint_path, payload, checkpoint_tmp_path)


def write_target_archive(
    df: pd.DataFrame,
    first_pos: int,
    rows_done: int,
    archive_dir: str,
    archive_prefix: str,
    checkpoint_delta_path: str,
    checkpoint_delta_tmp_path: str,
    target_family: str,
) -> int:
    os.makedirs(archive_dir, exist_ok=True)
    columns = _target_columns(target_family)
    slice_df = df.iloc[first_pos:rows_done][columns].dropna(subset=columns)
    if len(slice_df) > 0:
        fname = _target_archive_filename(
            archive_dir, archive_prefix, first_pos, rows_done - 1
        )
        _write_npz_file(
            fname,
            _archive_payload_from_slice(slice_df, target_family),
        )
        print(
            f"\nArchived {len(slice_df):,} {target_family.upper()} target rows "
            f"(pos {first_pos:,}–{rows_done - 1:,}) → {os.path.basename(fname)}"
        )
    _remove_files(checkpoint_delta_path, checkpoint_delta_tmp_path)
    return rows_done - 1


def save_target_checkpoint_delta(
    df: pd.DataFrame,
    archive_max_pos: int,
    rows_done: int,
    checkpoint_delta_path: str,
    checkpoint_delta_tmp_path: str,
    target_family: str,
):
    start = archive_max_pos + 1
    if start >= rows_done:
        return
    columns = _target_columns(target_family)
    ckpt_df = df.iloc[start:rows_done][columns].dropna(subset=columns)
    if len(ckpt_df) == 0:
        return
    if target_family == "sdxl":
        _write_npz_file(
            checkpoint_delta_path,
            _archive_payload_from_slice(ckpt_df, target_family),
            checkpoint_delta_tmp_path,
        )
        return
    ckpt_df.to_parquet(checkpoint_delta_tmp_path)
    with open(checkpoint_delta_tmp_path, "rb") as f:
        os.fsync(f.fileno())
    os.replace(checkpoint_delta_tmp_path, checkpoint_delta_path)


def load_target_archive_rows(archive_path: str, target_family: str) -> pd.DataFrame:
    with np.load(archive_path, allow_pickle=False) as data:
        if target_family == "sdxl":
            return pd.DataFrame(
                {
                    SDXL_PROMPT_COLUMN: list(data["prompt_embeds"]),
                    SDXL_POOLED_COLUMN: list(data["pooled_prompt_embeds"]),
                },
                index=data["indices"].astype(int),
            )
        return pd.DataFrame(
            {"clip_embedding": list(data["embeddings"])},
            index=data["indices"].astype(int),
        )


def load_target_checkpoint_delta_rows(
    checkpoint_path: str, target_family: str
) -> pd.DataFrame:
    if target_family == "sdxl":
        return load_target_archive_rows(checkpoint_path, target_family)
    delta = pd.read_parquet(checkpoint_path)
    delta["clip_embedding"] = delta["clip_embedding"].apply(
        lambda x: x.tolist() if hasattr(x, "tolist") else x
    )
    return delta


def promote_sdxl_checkpoint_delta_to_archive(
    checkpoint_path: str,
    checkpoint_tmp_path: str,
    archive_dir: str,
    archive_prefix: str,
    archive_start_pos: int,
) -> int:
    os.makedirs(archive_dir, exist_ok=True)
    with np.load(checkpoint_path, allow_pickle=False) as data:
        indices = data["indices"].astype(np.int64, copy=False)
        if len(indices) == 0:
            for path in [checkpoint_path, checkpoint_tmp_path]:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
            return archive_start_pos - 1

        archive_last_pos = int(indices.max())
        archive_path = _target_archive_filename(
            archive_dir,
            archive_prefix,
            archive_start_pos,
            archive_last_pos,
        )
        _write_npz_file(
            archive_path,
            {
                "indices": indices,
                "prompt_embeds": np.asarray(data["prompt_embeds"]),
                "pooled_prompt_embeds": np.asarray(data["pooled_prompt_embeds"]),
            },
        )
    _remove_files(checkpoint_path, checkpoint_tmp_path)

    print(
        "Promoted SDXL delta checkpoint to archive: "
        f"{os.path.basename(archive_path)}"
    )
    return archive_last_pos


def write_clip_archive(
    df: pd.DataFrame,
    first_pos: int,
    rows_done: int,
    archive_dir: str,
    archive_prefix: str,
    checkpoint_delta_path: str,
    checkpoint_delta_tmp_path: str,
) -> int:
    return write_target_archive(
        df,
        first_pos,
        rows_done,
        archive_dir,
        archive_prefix,
        checkpoint_delta_path,
        checkpoint_delta_tmp_path,
        target_family="sd",
    )


def save_clip_checkpoint_delta(
    df: pd.DataFrame,
    archive_max_pos: int,
    rows_done: int,
    checkpoint_delta_path: str,
    checkpoint_delta_tmp_path: str,
):
    save_target_checkpoint_delta(
        df,
        archive_max_pos,
        rows_done,
        checkpoint_delta_path,
        checkpoint_delta_tmp_path,
        target_family="sd",
    )


def save_target_errors(failed_rows, errors_path, errors_tmp_path):
    if not failed_rows:
        return
    new_df = pd.DataFrame(
        [
            {"original_index": idx, "prompt": prompt}
            for idx, prompt in failed_rows.items()
        ]
    )
    if os.path.exists(errors_path):
        try:
            existing = pd.read_parquet(errors_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["original_index"], keep="last")
        except Exception:
            combined = new_df
    else:
        combined = new_df
    _atomic_write(combined, errors_tmp_path, errors_path)
    print(
        f"\nTarget errors file updated: {len(combined):,} total failed rows → {errors_path}"
    )


# ==========================================
# MODEL LOADING & ENCODING
# ==========================================


def load_clip_model(target_family="sd"):
    target_family = get_target_family(target_family)
    target_spec = get_target_family_spec(target_family)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_label = f"GPU ({torch.cuda.get_device_name(0)})"
    else:
        # Fall back to CPU — give PyTorch every physical core.
        n_threads = os.cpu_count() or 4
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(max(1, n_threads // 4))
        device = torch.device("cpu")
        device_label = f"CPU ({os.cpu_count() or 4} threads)"

    # Use float16 on GPU (2x throughput, negligible quality loss for embeddings).
    # Keep float32 on CPU where float16 is typically slower.
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if target_family == "sdxl":
        print(
            f"Loading SDXL text encoders '{target_spec.model_id}' on {device_label} ({dtype})..."
        )
    else:
        print(
            f"Loading CLIP model '{target_spec.model_id or CLIP_MODEL_ID}' on {device_label} ({dtype})..."
        )
    _log_gpu_runtime(device)
    if target_family == "sdxl":
        tokenizer = (
            _load_hf_component(
                CLIPTokenizer,
                target_spec.tokenizer_id,
                token=HF_TOKEN,
            ),
            _load_hf_component(
                CLIPTokenizer,
                target_spec.tokenizer_2_id,
                token=HF_TOKEN,
            ),
        )
        text_encoder = (
            _load_hf_component(
                CLIPTextModel,
                target_spec.text_encoder_id,
                torch_dtype=dtype,
                token=HF_TOKEN,
            )
            .to(device)
            .eval(),
            _load_hf_component(
                CLIPTextModelWithProjection,
                target_spec.text_encoder_2_id,
                torch_dtype=dtype,
                token=HF_TOKEN,
            )
            .to(device)
            .eval(),
        )
        return tokenizer, text_encoder, device

    tokenizer = _load_hf_component(CLIPTokenizer, CLIP_MODEL_ID, token=HF_TOKEN)
    text_encoder = (
        _load_hf_component(
            CLIPTextModel,
            CLIP_MODEL_ID,
            torch_dtype=dtype,
            token=HF_TOKEN,
        )
        .to(device)
        .eval()
    )
    return tokenizer, text_encoder, device


def tokenize_batch(prompts, tokenizer, target_family="sd"):
    """Pure tokenization — no model involved, safe to run in a thread pool."""
    if target_family == "sdxl":
        tokenizer_one, tokenizer_two = tokenizer
        max_length = get_target_family_spec(target_family).sequence_length or 77
        return {
            "primary": tokenizer_one(
                prompts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ),
            "secondary": tokenizer_two(
                prompts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ),
        }

    result = tokenizer(
        prompts,
        padding="longest",
        max_length=LONG_CLIP_MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    # Round the padded length up to the next BUCKET_SIZE multiple so every
    # batch uses one of a small fixed set of tensor shapes.  ROCm (and MKL)
    # JIT-compiles a kernel per unique shape; without bucketing every batch
    # with a new max-length triggers a fresh compilation.
    cur_len = result["input_ids"].shape[1]
    bucket_len = min(
        ((cur_len + BUCKET_SIZE - 1) // BUCKET_SIZE) * BUCKET_SIZE,
        LONG_CLIP_MAX_LENGTH,
    )
    if bucket_len > cur_len:
        pad = bucket_len - cur_len
        result["input_ids"] = F.pad(
            result["input_ids"], (0, pad), value=tokenizer.pad_token_id
        )
        result["attention_mask"] = F.pad(result["attention_mask"], (0, pad), value=0)
        if "token_type_ids" in result:
            result["token_type_ids"] = F.pad(
                result["token_type_ids"], (0, pad), value=0
            )
    return result


@torch.inference_mode()
def encode_inputs(
    inputs,
    n_prompts,
    text_encoder,
    device,
    collect_timing=False,
    target_family="sd",
):
    """
    Run the target encoder forward pass on already-tokenized inputs.
    Returns a list of numpy payloads (or all-None on failure).
    """
    try:
        copy_in_start = perf_counter()
        if target_family == "sdxl":
            gpu_inputs = {
                key: {k: v.to(device, non_blocking=True) for k, v in value.items()}
                for key, value in inputs.items()
            }
        else:
            gpu_inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        if collect_timing:
            _sync_device(device)
        copy_in_s = perf_counter() - copy_in_start

        forward_start = perf_counter()
        if target_family == "sdxl":
            text_encoder_one, text_encoder_two = text_encoder
            outputs_one = text_encoder_one(
                **gpu_inputs["primary"],
                output_hidden_states=True,
                return_dict=True,
            )
            outputs_two = text_encoder_two(
                **gpu_inputs["secondary"],
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs = text_encoder(**gpu_inputs)
        if collect_timing:
            _sync_device(device)
        forward_s = perf_counter() - forward_start

        copy_out_start = perf_counter()
        if target_family == "sdxl":
            prompt_embeds = torch.cat(
                (outputs_one.hidden_states[-2], outputs_two.hidden_states[-2]), dim=-1
            )
            pooled_prompt_embeds = getattr(outputs_two, "text_embeds", None)
            if pooled_prompt_embeds is None:
                pooled_prompt_embeds = getattr(outputs_two, "pooler_output", None)
            if pooled_prompt_embeds is None:
                first_output = outputs_two[0]
                if first_output.ndim == 2:
                    pooled_prompt_embeds = first_output
                else:
                    pooled_prompt_embeds = outputs_two.last_hidden_state[:, -1]
            prompt_numpy = prompt_embeds.to(_sdxl_target_torch_dtype()).cpu().numpy()
            pooled_numpy = (
                pooled_prompt_embeds.to(_sdxl_target_torch_dtype()).cpu().numpy()
            )
            result = (prompt_numpy, pooled_numpy)
        else:
            # pooler_output: (batch, hidden_size) — projected [EOS] token, matches diffusers usage
            # .float() ensures float32 numpy arrays regardless of model dtype (float16 on GPU)
            embeddings = outputs.pooler_output.float().cpu().numpy()
            result = [embeddings[i] for i in range(n_prompts)]
        copy_out_s = perf_counter() - copy_out_start
        if collect_timing:
            return result, {
                "copy_in_s": copy_in_s,
                "forward_s": forward_s,
                "copy_out_s": copy_out_s,
            }
        return result
    except Exception as e:
        print(f"\nTarget encode failed: {e}")
        if collect_timing:
            return (None if target_family == "sdxl" else [None] * n_prompts), {
                "copy_in_s": 0.0,
                "forward_s": 0.0,
                "copy_out_s": 0.0,
            }
        return None if target_family == "sdxl" else [None] * n_prompts


# ==========================================
# MAIN PIPELINE
# ==========================================


def process_clip():
    target_family = get_target_family(
        os.getenv("CLIP_TARGET_FAMILY", os.getenv("TARGET_FAMILY", "sd"))
    )
    prompt_column = _requested_prompt_column()
    checkpoint_every = _checkpoint_every(target_family)
    archive_every = _archive_every(target_family)
    target_layout = get_target_layout(TARGET_OUTPUT_DIR, target_family)
    ensure_target_root(target_layout)
    write_target_manifest(
        target_layout.manifest_path,
        build_target_manifest(
            target_family,
            dtype=_target_archive_dtype_name(target_family),
            shard_size=archive_every,
            prompt_column=prompt_column,
        ),
    )

    active_target_paths = _resolve_active_target_paths(target_layout)
    direct_disk_target_storage = target_family == "sdxl"
    use_async_archive_writes = direct_disk_target_storage and TARGET_ARCHIVE_WRITERS > 0
    print(f"Target family: {target_family}")
    print(f"Qwen source root: {SOURCE_OUTPUT_DIR}")
    print(f"Target output root: {TARGET_OUTPUT_DIR}")
    if active_target_paths["using_legacy"]:
        print(f"Using legacy target storage: {active_target_paths['archive_dir']}")
    else:
        print(f"Using target storage: {active_target_paths['archive_dir']}")
    if direct_disk_target_storage:
        print("Archived SDXL targets will stay on disk during resume.")
    print(
        "Archive compression: "
        f"{'enabled' if TARGET_ARCHIVE_COMPRESS else 'disabled'}"
    )
    print(f"Archive dtype: {_target_archive_dtype_name(target_family)}")
    print(f"Batch size: {BATCH_SIZE:,} | tokenizer prefetch: {PREFETCH}")
    if direct_disk_target_storage:
        print(
            f"Archive writers: {TARGET_ARCHIVE_WRITERS} | inflight archive limit: {TARGET_ARCHIVE_MAX_INFLIGHT}"
        )
    print(
        f"Checkpoint every {checkpoint_every:,} rows | archive every {archive_every:,} rows"
    )

    # --- Load inputs ---
    qwen_df = load_qwen_embeddings()
    if qwen_df is None:
        print(
            "Error: No Qwen embeddings found. "
            "Run runner.py first (archives or checkpoint_latest.parquet required)."
        )
        raise SystemExit(1)

    print(f"Loading prompts: {DATASET_PATH}")
    prompts_df = pd.read_parquet(DATASET_PATH)

    # Align on the rows that runner.py has already finished, then free the
    # Qwen embeddings immediately — they're already safely in their own archives
    # and are not needed for CLIP processing.
    qwen_index = qwen_df.index
    del qwen_df
    df = prompts_df.loc[qwen_index].copy()
    del qwen_index, prompts_df
    if CLIP_MAX_ROWS > 0:
        df = df.iloc[:CLIP_MAX_ROWS].copy()
        print(f"Sample mode enabled: limiting CLIP run to {len(df):,} rows.")
    prompt_column = _resolve_prompt_column(df, prompt_column)
    _validate_source_manifest(prompt_column)
    target_columns = _target_columns(target_family)
    if not direct_disk_target_storage:
        _initialize_target_columns(df, target_family)
    total_rows = len(df)
    print(f"{total_rows:,} rows with Qwen embeddings available.")
    print(f"Using prompt column: {prompt_column}")

    # --- Discard any stale CLIP delta .tmp from a previous crash ---
    if os.path.exists(active_target_paths["checkpoint_delta_tmp_path"]):
        print("Removing stale CLIP delta .tmp left by a previous crash.")
        os.remove(active_target_paths["checkpoint_delta_tmp_path"])

    # --- Resume from CLIP two-tier checkpoint ---
    clip_archive_max_pos = get_clip_archive_max_pos(active_target_paths["archives"])
    clip_delta_max_pos = -1

    if not direct_disk_target_storage:
        for archive_path in active_target_paths["archives"]:
            arc_df = load_target_archive_rows(archive_path, target_family)
            _merge_target_rows(df, arc_df, target_family)
            del arc_df

    if os.path.exists(active_target_paths["checkpoint_delta_path"]):
        try:
            if direct_disk_target_storage:
                clip_archive_max_pos = promote_sdxl_checkpoint_delta_to_archive(
                    active_target_paths["checkpoint_delta_path"],
                    active_target_paths["checkpoint_delta_tmp_path"],
                    active_target_paths["archive_dir"],
                    active_target_paths["archive_prefix"],
                    clip_archive_max_pos + 1,
                )
                active_target_paths["archives"] = _list_target_archives(
                    active_target_paths["archive_dir"],
                    active_target_paths["archive_prefix"],
                )
            else:
                delta = load_target_checkpoint_delta_rows(
                    active_target_paths["checkpoint_delta_path"], target_family
                )
                _merge_target_rows(df, delta, target_family)
                clip_delta_max_pos = int(delta.index.max())
                del delta
        except Exception as e:
            print(f"CLIP delta checkpoint corrupted ({e}), discarding.")
            os.remove(active_target_paths["checkpoint_delta_path"])

    # Legacy fallback: old clip_checkpoint_latest.parquet
    if (
        target_family == "sd"
        and clip_archive_max_pos < 0
        and clip_delta_max_pos < 0
        and os.path.exists(CLIP_CHECKPOINT_PATH)
    ):
        print(f"Loading legacy CLIP checkpoint: {CLIP_CHECKPOINT_PATH}")
        try:
            ckpt = pd.read_parquet(CLIP_CHECKPOINT_PATH)
            df.loc[ckpt.index, "clip_embedding"] = ckpt["clip_embedding"]
            clip_delta_max_pos = int(ckpt.index.max())
            del ckpt
        except Exception as e:
            print(f"Legacy CLIP checkpoint corrupted ({e}), starting from scratch.")

    last_saved_pos = max(clip_archive_max_pos, clip_delta_max_pos)
    if last_saved_pos >= 0:
        if direct_disk_target_storage:
            start_i = last_saved_pos + 1
        else:
            start_i = ((last_saved_pos + 1) // BATCH_SIZE) * BATCH_SIZE
        print(
            f"Resuming from position {start_i:,} / {total_rows:,} "
            f"(archive up to {clip_archive_max_pos:,}, delta up to {clip_delta_max_pos:,})"
        )
    else:
        start_i = 0

    if start_i >= total_rows:
        print("All rows already have CLIP embeddings!")
        if direct_disk_target_storage:
            return
        remaining_start = clip_archive_max_pos + 1
        if remaining_start < total_rows:
            try:
                write_target_archive(
                    df,
                    remaining_start,
                    total_rows,
                    active_target_paths["archive_dir"],
                    active_target_paths["archive_prefix"],
                    active_target_paths["checkpoint_delta_path"],
                    active_target_paths["checkpoint_delta_tmp_path"],
                    target_family,
                )
            except Exception as arc_err:
                print(f"\nWARNING: final CLIP archive failed: {arc_err}")
        return

    print(f"Processing {total_rows - start_i:,} remaining rows...")

    tokenizer, text_encoder, device = load_clip_model(target_family)
    failed_rows: dict[int, str] = {}
    if TIMING_BATCHES != 0:
        timing_scope = "all" if TIMING_BATCHES < 0 else str(TIMING_BATCHES)
        print(
            f"Timing enabled: logging every {TIMING_EVERY} batch(es) for first {timing_scope} batch(es)."
        )

    # Build the list of (positional_start, batch_slice) tuples up front
    batch_starts = list(range(start_i, total_rows, BATCH_SIZE))

    # ---------------------------------------------------------------
    # Pipeline: a background ThreadPoolExecutor tokenizes PREFETCH
    # batches ahead so the forward-pass thread never waits for I/O.
    # Tokenizers release the GIL during their C extensions, so real
    # parallelism is achieved without multiprocessing overhead.
    # ---------------------------------------------------------------
    target_column_positions = None
    if not direct_disk_target_storage:
        target_column_positions = {
            column: df.columns.get_loc(column) for column in target_columns
        }
    _prompt_col = df.columns.get_loc(prompt_column)
    writer_pool = (
        ThreadPoolExecutor(max_workers=TARGET_ARCHIVE_WRITERS)
        if use_async_archive_writes
        else None
    )
    pending_archive_writes: deque = deque()
    pending_sdxl_payload = (
        _new_sdxl_payload_buffer() if direct_disk_target_storage else None
    )
    durable_archive_max_pos = clip_archive_max_pos
    submitted_archive_max_pos = clip_archive_max_pos

    def _tokenize(pos):
        """Tokenize one batch; returns (pos, batch_slice, inputs)."""
        batch_slice = df.iloc[pos : pos + BATCH_SIZE]
        prompts = batch_slice[prompt_column].tolist()
        inputs = tokenize_batch(prompts, tokenizer, target_family)
        return pos, batch_slice, inputs

    pbar = tqdm(total=total_rows - start_i, desc="CLIP Embeddings")
    i = start_i
    rows_done = start_i
    batch_num = 0
    completed_all_rows = False
    try:
        with ThreadPoolExecutor(max_workers=PREFETCH) as pool:
            # Seed the queue with the first PREFETCH futures
            future_queue = [
                pool.submit(_tokenize, pos) for pos in batch_starts[:PREFETCH]
            ]
            next_batch_idx = PREFETCH  # index into batch_starts for the next submission

            for future in future_queue:
                batch_num += 1
                log_timing = _should_log_timing(batch_num)
                batch_start = perf_counter()

                wait_start = perf_counter()
                i, batch_slice, inputs = future.result()
                wait_s = perf_counter() - wait_start
                bucket_len = _target_input_length(inputs, target_family)

                # Submit the next batch tokenization immediately so it runs
                # in the background while we do the forward pass below.
                if next_batch_idx < len(batch_starts):
                    future_queue.append(
                        pool.submit(_tokenize, batch_starts[next_batch_idx])
                    )
                    next_batch_idx += 1

                if log_timing:
                    embeddings, encode_timing = encode_inputs(
                        inputs,
                        len(batch_slice),
                        text_encoder,
                        device,
                        collect_timing=True,
                        target_family=target_family,
                    )
                else:
                    embeddings = encode_inputs(
                        inputs,
                        len(batch_slice),
                        text_encoder,
                        device,
                        target_family=target_family,
                    )
                    encode_timing = {
                        "copy_in_s": 0.0,
                        "forward_s": 0.0,
                        "copy_out_s": 0.0,
                    }

                write_start = perf_counter()
                batch_end = i + len(batch_slice)
                if direct_disk_target_storage:
                    batch_indices = df.index[i:batch_end].to_numpy(
                        dtype=np.int64, copy=False
                    )
                    batch_positions = np.arange(i, batch_end, dtype=np.int64)
                    if embeddings is None:
                        for row_pos, original_index in enumerate(batch_indices):
                            failed_rows[int(original_index)] = df.iat[
                                i + row_pos, _prompt_col
                            ]
                    else:
                        prompt_batch, pooled_batch = embeddings
                        _append_sdxl_payload_buffer(
                            pending_sdxl_payload,
                            batch_positions,
                            batch_indices.copy(),
                            prompt_batch,
                            pooled_batch,
                        )
                else:
                    for j, emb in enumerate(embeddings):
                        if emb is not None:
                            _store_target_value(
                                df,
                                i + j,
                                emb,
                                target_family,
                                target_column_positions,
                            )
                        else:
                            failed_rows[int(df.index[i + j])] = df.iat[
                                i + j, _prompt_col
                            ]
                write_s = perf_counter() - write_start

                pbar.update(len(batch_slice))

                rows_done = min(batch_end, total_rows)
                ckpt_s = 0.0
                archive_s = 0.0
                if direct_disk_target_storage:
                    try:
                        archive_start = perf_counter()
                        completed_pos = _drain_completed_target_archives(
                            pending_archive_writes
                        )
                        if completed_pos is not None:
                            durable_archive_max_pos = completed_pos

                        while (
                            rows_done - (submitted_archive_max_pos + 1) >= archive_every
                        ):
                            archive_first_pos = submitted_archive_max_pos + 1
                            archive_last_pos = archive_first_pos + archive_every - 1
                            while (
                                writer_pool is not None
                                and len(pending_archive_writes)
                                >= TARGET_ARCHIVE_MAX_INFLIGHT
                            ):
                                completed_pos = _drain_completed_target_archives(
                                    pending_archive_writes,
                                    block=True,
                                )
                                if completed_pos is not None:
                                    durable_archive_max_pos = completed_pos
                            payload = _pop_sdxl_payload_buffer(
                                pending_sdxl_payload,
                                archive_last_pos,
                            )
                            submitted_archive_max_pos = archive_last_pos
                            if payload is None:
                                durable_archive_max_pos = archive_last_pos
                                continue
                            archive_path = _target_archive_filename(
                                active_target_paths["archive_dir"],
                                active_target_paths["archive_prefix"],
                                archive_first_pos,
                                archive_last_pos,
                            )
                            if writer_pool is not None:
                                future = writer_pool.submit(
                                    _write_target_archive_payload,
                                    archive_path,
                                    payload,
                                )
                                pending_archive_writes.append(
                                    {
                                        "future": future,
                                        "first_pos": archive_first_pos,
                                        "last_pos": archive_last_pos,
                                        "archive_path": archive_path,
                                        "payload": payload,
                                    }
                                )
                            else:
                                _write_target_archive_payload(archive_path, payload)
                                print(
                                    f"\nArchived {len(payload['indices']):,} SDXL target rows "
                                    f"(pos {archive_first_pos:,}–{archive_last_pos:,}) → {os.path.basename(archive_path)}"
                                )
                                durable_archive_max_pos = archive_last_pos
                        archive_s = perf_counter() - archive_start
                    except Exception as arc_err:
                        print(f"\nWARNING: CLIP archive failed: {arc_err}")
                    if rows_done % checkpoint_every < BATCH_SIZE:
                        try:
                            ckpt_start = perf_counter()
                            _save_sdxl_checkpoint_payload(
                                _sdxl_checkpoint_payload(
                                    pending_archive_writes,
                                    pending_sdxl_payload,
                                ),
                                active_target_paths["checkpoint_delta_path"],
                                active_target_paths["checkpoint_delta_tmp_path"],
                            )
                            ckpt_s = perf_counter() - ckpt_start
                            print(
                                f"\nCLIP checkpoint saved ({min(rows_done, total_rows):,} rows)"
                            )
                        except Exception as ckpt_err:
                            print(f"\nWARNING: CLIP checkpoint failed: {ckpt_err}")
                else:
                    if rows_done - (clip_archive_max_pos + 1) >= archive_every:
                        try:
                            archive_start = perf_counter()
                            archive_first_pos = clip_archive_max_pos + 1
                            clip_archive_max_pos = write_target_archive(
                                df,
                                archive_first_pos,
                                rows_done,
                                active_target_paths["archive_dir"],
                                active_target_paths["archive_prefix"],
                                active_target_paths["checkpoint_delta_path"],
                                active_target_paths["checkpoint_delta_tmp_path"],
                                target_family,
                            )
                            _clear_target_rows(
                                df,
                                archive_first_pos,
                                clip_archive_max_pos + 1,
                                target_family,
                            )
                            archive_s = perf_counter() - archive_start
                        except Exception as arc_err:
                            print(f"\nWARNING: CLIP archive failed: {arc_err}")
                    if rows_done % checkpoint_every < BATCH_SIZE:
                        try:
                            ckpt_start = perf_counter()
                            save_target_checkpoint_delta(
                                df,
                                clip_archive_max_pos,
                                rows_done,
                                active_target_paths["checkpoint_delta_path"],
                                active_target_paths["checkpoint_delta_tmp_path"],
                                target_family,
                            )
                            ckpt_s = perf_counter() - ckpt_start
                            print(
                                f"\nCLIP checkpoint saved ({min(rows_done, total_rows):,} rows)"
                            )
                        except Exception as ckpt_err:
                            print(f"\nWARNING: CLIP checkpoint failed: {ckpt_err}")

                if log_timing:
                    batch_s = perf_counter() - batch_start
                    print(
                        "TIMING "
                        f"batch={batch_num} pos={i:,} size={len(batch_slice)} bucket={bucket_len} "
                        f"wait={wait_s:.3f}s copy_in={encode_timing['copy_in_s']:.3f}s "
                        f"forward={encode_timing['forward_s']:.3f}s "
                        f"copy_out={encode_timing['copy_out_s']:.3f}s write={write_s:.3f}s "
                        f"ckpt={ckpt_s:.3f}s archive={archive_s:.3f}s total={batch_s:.3f}s "
                        f"rate={len(batch_slice) / batch_s:.2f}it/s"
                    )

        completed_all_rows = True

    finally:
        try:
            pbar.close()
            save_target_errors(
                failed_rows,
                active_target_paths["errors_path"],
                active_target_paths["errors_tmp_path"],
            )
            if direct_disk_target_storage:
                try:
                    completed_pos = _drain_completed_target_archives(
                        pending_archive_writes
                    )
                    if completed_pos is not None:
                        durable_archive_max_pos = completed_pos

                    if completed_all_rows and rows_done >= total_rows:
                        final_first_pos = submitted_archive_max_pos + 1
                        final_last_pos = total_rows - 1
                        if final_first_pos <= final_last_pos:
                            while (
                                writer_pool is not None
                                and len(pending_archive_writes)
                                >= TARGET_ARCHIVE_MAX_INFLIGHT
                            ):
                                completed_pos = _drain_completed_target_archives(
                                    pending_archive_writes,
                                    block=True,
                                )
                                if completed_pos is not None:
                                    durable_archive_max_pos = completed_pos
                            payload = _pop_sdxl_payload_buffer(
                                pending_sdxl_payload,
                                final_last_pos,
                            )
                            submitted_archive_max_pos = final_last_pos
                            if payload is None:
                                durable_archive_max_pos = final_last_pos
                            else:
                                archive_path = _target_archive_filename(
                                    active_target_paths["archive_dir"],
                                    active_target_paths["archive_prefix"],
                                    final_first_pos,
                                    final_last_pos,
                                )
                                if writer_pool is not None:
                                    future = writer_pool.submit(
                                        _write_target_archive_payload,
                                        archive_path,
                                        payload,
                                    )
                                    pending_archive_writes.append(
                                        {
                                            "future": future,
                                            "first_pos": final_first_pos,
                                            "last_pos": final_last_pos,
                                            "archive_path": archive_path,
                                            "payload": payload,
                                        }
                                    )
                                else:
                                    _write_target_archive_payload(archive_path, payload)
                                    print(
                                        f"\nArchived {len(payload['indices']):,} SDXL target rows "
                                        f"(pos {final_first_pos:,}–{final_last_pos:,}) → {os.path.basename(archive_path)}"
                                    )
                                    durable_archive_max_pos = final_last_pos

                        while pending_archive_writes:
                            completed_pos = _drain_completed_target_archives(
                                pending_archive_writes,
                                block=True,
                            )
                            if completed_pos is not None:
                                durable_archive_max_pos = completed_pos
                        _remove_files(
                            active_target_paths["checkpoint_delta_path"],
                            active_target_paths["checkpoint_delta_tmp_path"],
                        )
                    else:
                        checkpoint_payload = _sdxl_checkpoint_payload(
                            pending_archive_writes,
                            pending_sdxl_payload,
                        )
                        _save_sdxl_checkpoint_payload(
                            checkpoint_payload,
                            active_target_paths["checkpoint_delta_path"],
                            active_target_paths["checkpoint_delta_tmp_path"],
                        )
                        print(
                            f"\nCLIP progress saved at position {rows_done:,}. Re-run to resume."
                        )
                        return
                except Exception as e:
                    print(f"\nWARNING: CLIP shutdown save failed: {e}")
                    return
            else:
                rows_done = min(i + BATCH_SIZE, total_rows)
                if rows_done < total_rows:
                    try:
                        if rows_done - (clip_archive_max_pos + 1) >= archive_every:
                            archive_first_pos = clip_archive_max_pos + 1
                            clip_archive_max_pos = write_target_archive(
                                df,
                                archive_first_pos,
                                rows_done,
                                active_target_paths["archive_dir"],
                                active_target_paths["archive_prefix"],
                                active_target_paths["checkpoint_delta_path"],
                                active_target_paths["checkpoint_delta_tmp_path"],
                                target_family,
                            )
                            _clear_target_rows(
                                df,
                                archive_first_pos,
                                clip_archive_max_pos + 1,
                                target_family,
                            )
                        save_target_checkpoint_delta(
                            df,
                            clip_archive_max_pos,
                            rows_done,
                            active_target_paths["checkpoint_delta_path"],
                            active_target_paths["checkpoint_delta_tmp_path"],
                            target_family,
                        )
                        print(
                            f"\nCLIP progress saved at position {rows_done:,}. Re-run to resume."
                        )
                    except Exception as e:
                        print(f"\nWARNING: CLIP shutdown save failed: {e}")
                    return
        finally:
            if writer_pool is not None:
                writer_pool.shutdown(wait=True)

    if direct_disk_target_storage:
        print(f"\nPipeline complete! {total_rows:,} rows processed.")
        if failed_rows:
            print(
                f"{len(failed_rows):,} rows failed — see {active_target_paths['errors_path']} to retry."
            )
        return

    # Write the final CLIP archive for any remaining unarchived rows
    remaining_start = clip_archive_max_pos + 1
    if remaining_start < total_rows:
        try:
            write_target_archive(
                df,
                remaining_start,
                total_rows,
                active_target_paths["archive_dir"],
                active_target_paths["archive_prefix"],
                active_target_paths["checkpoint_delta_path"],
                active_target_paths["checkpoint_delta_tmp_path"],
                target_family,
            )
        except Exception as arc_err:
            print(f"\nWARNING: final CLIP archive failed: {arc_err}")

    print(f"\nPipeline complete! {total_rows:,} rows processed.")
    if failed_rows:
        print(
            f"{len(failed_rows):,} rows failed — see {active_target_paths['errors_path']} to retry."
        )


if __name__ == "__main__":
    process_clip()
