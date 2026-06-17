#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# v8 trains the sequence-aware SDXL resampler against token-sequence Qwen
# archives. Defaults are set for the full-dataset streaming path so the 1M-row
# roots are not materialized in host RAM.
export TARGET_OUTPUT_DIR="${TARGET_OUTPUT_DIR:-embedded_chunks/clip_archive}"
export QWEN_OUTPUT_DIR="${QWEN_OUTPUT_DIR:-embedded_chunks/qwen3.5-27b}"
export TRAIN_TARGET_FAMILY="${TRAIN_TARGET_FAMILY:-sdxl}"
export CPU_THREADS="${CPU_THREADS:-7}"
export TRAIN_ARCHIVE_THREADS="${TRAIN_ARCHIVE_THREADS:-2}"
export TRAIN_ARCHIVE_MODE="${TRAIN_ARCHIVE_MODE:-stream}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
export TRAIN_VAL_SPLIT="${TRAIN_VAL_SPLIT:-0.05}"
export TRAIN_VAL_MAX_STEPS="${TRAIN_VAL_MAX_STEPS:-0}"
export TRAIN_VAL_EVERY_N_EPOCHS="${TRAIN_VAL_EVERY_N_EPOCHS:-1}"
export TRAIN_STREAM_VAL_CACHE="${TRAIN_STREAM_VAL_CACHE:-1}"
export TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-0}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-0}"
export TRAIN_MAX_LR="${TRAIN_MAX_LR:-1e-4}"
export TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-1e-5}"
export TRAIN_PCT_START="${TRAIN_PCT_START:-0.10}"
export TRAIN_GRAD_CLIP="${TRAIN_GRAD_CLIP:-1.0}"
export TRAIN_HIDDEN_DIM="${TRAIN_HIDDEN_DIM:-4096}"
export TRAIN_PROGRESS_EVERY="${TRAIN_PROGRESS_EVERY:-50}"
export TRAIN_PROGRESS_SECONDS="${TRAIN_PROGRESS_SECONDS:-30}"
export TRAIN_TIMING="${TRAIN_TIMING:-0}"
export RUN_LABEL="${RUN_LABEL:-stream_v8_resampler_tokens_b512}"
export TRAIN_V8_AUTO_CAP_ALIGNED_ROWS="${TRAIN_V8_AUTO_CAP_ALIGNED_ROWS:-1}"

if [[ "$TRAIN_ARCHIVE_MODE" != "stream" && "$TRAIN_ARCHIVE_MODE" != "streaming" ]]; then
	echo "train_v8.sh defaults to the full-dataset streaming path." >&2
	echo "Set TRAIN_ARCHIVE_MODE=stream or streaming; got '$TRAIN_ARCHIVE_MODE'." >&2
	exit 1
fi

# Explicit cold start.
export TRAIN_WARM_START_PATH=""
export TRAIN_WARM_START_STRICT="${TRAIN_WARM_START_STRICT:-1}"
export TRAIN_AUTO_RESUME="${TRAIN_AUTO_RESUME:-1}"

# The throughput probes used raw target space. Standardization remains opt-in;
# keep standardized archive caching off by default because the full cache is very
# large on disk.
export TRAIN_STANDARDIZE_EMBEDDINGS="${TRAIN_STANDARDIZE_EMBEDDINGS:-0}"
export TRAIN_STANDARDIZATION_EPS="${TRAIN_STANDARDIZATION_EPS:-1e-6}"
export TRAIN_STANDARDIZATION_THREADS="${TRAIN_STANDARDIZATION_THREADS:-$CPU_THREADS}"
export TRAIN_CACHE_STANDARDIZED_ARCHIVES="${TRAIN_CACHE_STANDARDIZED_ARCHIVES:-0}"
export TRAIN_STANDARDIZED_ARCHIVE_CACHE_DIR="${TRAIN_STANDARDIZED_ARCHIVE_CACHE_DIR:-}"
export TRAIN_STANDARDIZED_ARCHIVE_COMPRESS="${TRAIN_STANDARDIZED_ARCHIVE_COMPRESS:-0}"
export TRAIN_STANDARDIZED_ARCHIVE_CACHE_DTYPE="${TRAIN_STANDARDIZED_ARCHIVE_CACHE_DTYPE:-source}"

export TRAIN_SDXL_PROJECTOR_ARCH="${TRAIN_SDXL_PROJECTOR_ARCH:-resampler}"
export TRAIN_SDXL_PROMPT_LOSS_WEIGHT="${TRAIN_SDXL_PROMPT_LOSS_WEIGHT:-1.0}"
export TRAIN_SDXL_POOLED_LOSS_WEIGHT="${TRAIN_SDXL_POOLED_LOSS_WEIGHT:-1.0}"
export TRAIN_SDXL_POINTWISE_LOSS="${TRAIN_SDXL_POINTWISE_LOSS:-mse}"
export TRAIN_SDXL_HUBER_DELTA="${TRAIN_SDXL_HUBER_DELTA:-1.0}"
export TRAIN_SDXL_PROMPT_COSINE_WEIGHT="${TRAIN_SDXL_PROMPT_COSINE_WEIGHT:-0.10}"
export TRAIN_SDXL_POOLED_COSINE_WEIGHT="${TRAIN_SDXL_POOLED_COSINE_WEIGHT:-0.05}"
export TRAIN_SDXL_PROMPT_NORM_WEIGHT="${TRAIN_SDXL_PROMPT_NORM_WEIGHT:-0.10}"
export TRAIN_SDXL_POOLED_NORM_WEIGHT="${TRAIN_SDXL_POOLED_NORM_WEIGHT:-0.05}"
export TRAIN_SDXL_PROMPT_NORM_MATCH_WEIGHT="${TRAIN_SDXL_PROMPT_NORM_MATCH_WEIGHT:-0.0}"
export TRAIN_SDXL_POOLED_NORM_MATCH_WEIGHT="${TRAIN_SDXL_POOLED_NORM_MATCH_WEIGHT:-0.0}"
export TRAIN_SDXL_MONITOR_METRIC="${TRAIN_SDXL_MONITOR_METRIC:-composite}"
export TRAIN_SDXL_MONITOR_NORM_WEIGHT="${TRAIN_SDXL_MONITOR_NORM_WEIGHT:-0.10}"
export TRAIN_SDXL_MONITOR_STD_WEIGHT="${TRAIN_SDXL_MONITOR_STD_WEIGHT:-0.05}"
export TRAIN_SDXL_MONITOR_PROMPT_NORM_MATCH_WEIGHT="${TRAIN_SDXL_MONITOR_PROMPT_NORM_MATCH_WEIGHT:-0.0}"
export TRAIN_SDXL_MONITOR_POOLED_NORM_MATCH_WEIGHT="${TRAIN_SDXL_MONITOR_POOLED_NORM_MATCH_WEIGHT:-0.0}"
export TRAIN_SDXL_RESAMPLER_DEPTH="${TRAIN_SDXL_RESAMPLER_DEPTH:-2}"
export TRAIN_SDXL_RESAMPLER_HEADS="${TRAIN_SDXL_RESAMPLER_HEADS:-8}"
export TRAIN_SDXL_RESAMPLER_FF_MULT="${TRAIN_SDXL_RESAMPLER_FF_MULT:-4}"
export TRAIN_SDXL_RESAMPLER_POOLED_QUERIES="${TRAIN_SDXL_RESAMPLER_POOLED_QUERIES:-1}"
export TRAIN_SDXL_PROMPT_HEAD_DIM="${TRAIN_SDXL_PROMPT_HEAD_DIM:-512}"
export TRAIN_SDXL_POOLED_HEAD_DIM="${TRAIN_SDXL_POOLED_HEAD_DIM:-2048}"

PREFLIGHT_ENV="$(mktemp)"
QWEN_OUTPUT_DIR="$QWEN_OUTPUT_DIR" \
TARGET_OUTPUT_DIR="$TARGET_OUTPUT_DIR" \
TRAIN_VAL_SPLIT="$TRAIN_VAL_SPLIT" \
TRAIN_VAL_MAX_STEPS="$TRAIN_VAL_MAX_STEPS" \
TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
TRAIN_MAX_SAMPLES="$TRAIN_MAX_SAMPLES" \
TRAIN_V8_AUTO_CAP_ALIGNED_ROWS="$TRAIN_V8_AUTO_CAP_ALIGNED_ROWS" \
PREFLIGHT_ENV="$PREFLIGHT_ENV" \
uv run python - <<'PY'
import json
import math
import os
import re
from pathlib import Path

import numpy as np

qwen_root = Path(os.environ["QWEN_OUTPUT_DIR"])
target_root = Path(os.environ["TARGET_OUTPUT_DIR"])
qwen_archive_dir = qwen_root / "archive"
clip_archive_dir = target_root
if not any(clip_archive_dir.glob("target_archive_*.npz")):
	nested = target_root / "targets" / "sdxl" / "archive"
	if any(nested.glob("target_archive_*.npz")):
		clip_archive_dir = nested

manifest_path = qwen_root / "source_manifest.json"
if not manifest_path.exists():
	raise SystemExit(
		f"Missing {manifest_path}. Run bash extract_qwen_no_pooling.sh against "
		"a sequence-preserving proxy before launching v8."
	)
with open(manifest_path, "r", encoding="utf-8") as handle:
	manifest = json.load(handle)
if manifest.get("embedding_format") != "token_sequence":
	raise SystemExit(
		"v8 requires token-sequence Qwen archives. "
		f"{manifest_path} reports embedding_format={manifest.get('embedding_format')!r}."
	)

def list_archives(path: Path, prefix: str):
	pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{7}})_(\d{{7}})\.npz$")
	archives = [child for child in path.iterdir() if child.is_file() and pattern.match(child.name)]
	archives.sort(key=lambda child: int(pattern.match(child.name).group(1)))
	if not archives:
		raise SystemExit(f"No {prefix}_*.npz files found in {path}")
	return archives

def load_indices(archives):
	parts = []
	rows_per_archive = []
	for archive in archives:
		with np.load(archive, allow_pickle=False) as data:
			indices = data["indices"].astype(np.int64, copy=False)
			parts.append(indices.copy())
			rows_per_archive.append(len(indices))
	return np.concatenate(parts), rows_per_archive

qwen_archives = list_archives(qwen_archive_dir, "archive")
clip_archives = list_archives(clip_archive_dir, "target_archive")
qwen_indices, qwen_rows = load_indices(qwen_archives)
clip_indices, clip_rows = load_indices(clip_archives)
common_total = min(len(qwen_indices), len(clip_indices))
requested_max_samples = int(os.environ["TRAIN_MAX_SAMPLES"])
auto_cap = os.environ["TRAIN_V8_AUTO_CAP_ALIGNED_ROWS"] == "1"
effective_max_samples = requested_max_samples
if len(qwen_indices) != len(clip_indices):
	if requested_max_samples <= 0:
		if not auto_cap:
			raise SystemExit(
				"Qwen and CLIP archive row counts differ. Set TRAIN_MAX_SAMPLES "
				f"to the common aligned prefix ({common_total}) or set "
				"TRAIN_V8_AUTO_CAP_ALIGNED_ROWS=1."
			)
		effective_max_samples = common_total
	elif requested_max_samples > common_total:
		raise SystemExit(
			"TRAIN_MAX_SAMPLES exceeds the common aligned prefix: "
			f"requested={requested_max_samples:,}, common={common_total:,}."
		)

aligned_rows = common_total
if effective_max_samples > 0:
	aligned_rows = min(aligned_rows, effective_max_samples)
if aligned_rows <= 0:
	raise SystemExit("No aligned rows available for v8 streaming training.")
if not np.array_equal(qwen_indices[:aligned_rows], clip_indices[:aligned_rows]):
	mismatch = int(np.flatnonzero(qwen_indices[:aligned_rows] != clip_indices[:aligned_rows])[0])
	raise SystemExit(
		"Qwen and CLIP archive indices do not align at row "
		f"{mismatch}: qwen={int(qwen_indices[mismatch])}, clip={int(clip_indices[mismatch])}."
	)

val_fraction = float(os.environ["TRAIN_VAL_SPLIT"])
val_max_steps = int(os.environ["TRAIN_VAL_MAX_STEPS"])
batch_size = int(os.environ["TRAIN_BATCH_SIZE"])
train_rows = aligned_rows - int(aligned_rows * val_fraction) if val_fraction > 0 else aligned_rows
val_rows = aligned_rows - train_rows
train_steps = math.ceil(train_rows / batch_size)
val_steps = math.ceil(val_rows / batch_size) if val_rows > 0 else 0
val_run_steps = min(val_steps, val_max_steps) if val_max_steps > 0 else val_steps
val_run_rows = min(val_rows, val_run_steps * batch_size) if val_run_steps else 0
print("v8 streaming preflight")
print(f"Qwen archives: {qwen_archive_dir} | files={len(qwen_archives)} | rows={len(qwen_indices):,} | shard_rows={min(qwen_rows):,}-{max(qwen_rows):,}")
print(f"CLIP archives: {clip_archive_dir} | files={len(clip_archives)} | rows={len(clip_indices):,} | shard_rows={min(clip_rows):,}-{max(clip_rows):,}")
if effective_max_samples != requested_max_samples:
	print(f"Auto-capping TRAIN_MAX_SAMPLES to aligned common prefix: {effective_max_samples:,}")
print(f"Aligned rows: {aligned_rows:,} | train_rows~{train_rows:,} | val_rows~{val_rows:,}")
print(f"Steps per epoch: train~{train_steps:,}" + (f" | val~{val_run_steps:,}/{val_steps:,}" if val_steps else ""))
if val_steps and val_max_steps > 0:
	print(f"Validation is capped to ~{val_run_rows:,} rows per epoch via TRAIN_VAL_MAX_STEPS={val_max_steps}.")

with open(os.environ["PREFLIGHT_ENV"], "w", encoding="utf-8") as handle:
	if effective_max_samples != requested_max_samples:
		handle.write(f"export TRAIN_MAX_SAMPLES={effective_max_samples}\n")
PY

if [[ -s "$PREFLIGHT_ENV" ]]; then
	# shellcheck disable=SC1090
	source "$PREFLIGHT_ENV"
fi
rm -f "$PREFLIGHT_ENV"

if [[ "${TRAIN_V8_DRY_RUN:-0}" == "1" || "${TRAIN_STREAM_DRY_RUN:-0}" == "1" ]]; then
	echo "Dry run requested; exiting before training."
	exit 0
fi

exec "$ROOT_DIR/train_sdxl_optimized.sh"