#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Generate native SDXL dual-text-encoder targets for the sequence-aware
# Qwen resampler path. The targets are written under:
#   $TARGET_OUTPUT_DIR

if [[ -z "${QWEN_OUTPUT_DIR:-}" ]]; then
    if [[ -f "embedded_chunks/qwen3.5-27b/source_manifest.json" ]]; then
        export QWEN_OUTPUT_DIR="embedded_chunks/qwen3.5-27b"
    elif [[ -f "embedded_chunks_no_pooling/source_manifest.json" ]]; then
        export QWEN_OUTPUT_DIR="embedded_chunks_no_pooling"
    else
        export QWEN_OUTPUT_DIR="embedded_chunks/qwen3.5-27b"
    fi
else
    export QWEN_OUTPUT_DIR
fi

export TARGET_OUTPUT_DIR="${TARGET_OUTPUT_DIR:-embedded_chunks/clip_archive}"
export CLIP_TARGET_FAMILY="${CLIP_TARGET_FAMILY:-sdxl}"
export TARGET_FAMILY="${TARGET_FAMILY:-$CLIP_TARGET_FAMILY}"

if [[ "$CLIP_TARGET_FAMILY" != "sdxl" ]]; then
    echo "extract_clip.sh is for SDXL targets; got CLIP_TARGET_FAMILY=$CLIP_TARGET_FAMILY" >&2
    exit 1
fi

MANIFEST_PATH="$QWEN_OUTPUT_DIR/source_manifest.json"
if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "Missing $MANIFEST_PATH" >&2
    echo "Run bash extract_qwen_no_pooling.sh first, or set QWEN_OUTPUT_DIR to the raw Qwen archive root." >&2
    exit 1
fi

QWEN_SOURCE_MANIFEST="$MANIFEST_PATH" python3 - <<'PY'
import json
import os

manifest_path = os.environ["QWEN_SOURCE_MANIFEST"]
with open(manifest_path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)

embedding_format = payload.get("embedding_format")
if embedding_format != "token_sequence":
    raise SystemExit(
        "SDXL resampler target extraction expects token-sequence Qwen archives. "
        f"{manifest_path} reports embedding_format={embedding_format!r}."
    )
PY

export SDXL_TARGET_DTYPE="${SDXL_TARGET_DTYPE:-float16}"
export TARGET_ARCHIVE_COMPRESS="${TARGET_ARCHIVE_COMPRESS:-0}"
export TARGET_ARCHIVE_WRITERS="${TARGET_ARCHIVE_WRITERS:-7}"
export TARGET_ARCHIVE_MAX_INFLIGHT="${TARGET_ARCHIVE_MAX_INFLIGHT:-$TARGET_ARCHIVE_WRITERS}"
export CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-2560}"
export CLIP_PREFETCH="${CLIP_PREFETCH:-1}"
export SDXL_CLIP_ARCHIVE_EVERY="${SDXL_CLIP_ARCHIVE_EVERY:-$((5 * CLIP_BATCH_SIZE))}"
export SDXL_CHECKPOINT_EVERY="${SDXL_CHECKPOINT_EVERY:-$((20 * CLIP_BATCH_SIZE))}"

echo "Starting SDXL target extraction"
echo "Qwen source root: $QWEN_OUTPUT_DIR"
echo "Target output root: $TARGET_OUTPUT_DIR"
echo "Target family: $CLIP_TARGET_FAMILY"
echo "Archive dtype: $SDXL_TARGET_DTYPE"
echo "Archive compression: $TARGET_ARCHIVE_COMPRESS"
echo "Archive rows: $SDXL_CLIP_ARCHIVE_EVERY"
echo "Checkpoint rows: $SDXL_CHECKPOINT_EVERY"
echo "Batch size: $CLIP_BATCH_SIZE | tokenizer prefetch: $CLIP_PREFETCH"
echo "Archive writers: $TARGET_ARCHIVE_WRITERS | inflight archive limit: $TARGET_ARCHIVE_MAX_INFLIGHT"

exec env \
    QWEN_OUTPUT_DIR="$QWEN_OUTPUT_DIR" \
    TARGET_OUTPUT_DIR="$TARGET_OUTPUT_DIR" \
    CLIP_TARGET_FAMILY="$CLIP_TARGET_FAMILY" \
    TARGET_FAMILY="$TARGET_FAMILY" \
    SDXL_TARGET_DTYPE="$SDXL_TARGET_DTYPE" \
    TARGET_ARCHIVE_COMPRESS="$TARGET_ARCHIVE_COMPRESS" \
    TARGET_ARCHIVE_WRITERS="$TARGET_ARCHIVE_WRITERS" \
    TARGET_ARCHIVE_MAX_INFLIGHT="$TARGET_ARCHIVE_MAX_INFLIGHT" \
    CLIP_BATCH_SIZE="$CLIP_BATCH_SIZE" \
    CLIP_PREFETCH="$CLIP_PREFETCH" \
    SDXL_CLIP_ARCHIVE_EVERY="$SDXL_CLIP_ARCHIVE_EVERY" \
    SDXL_CHECKPOINT_EVERY="$SDXL_CHECKPOINT_EVERY" \
    uv run python clip_runner.py