#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Re-extract Qwen embeddings as token sequences for the resampler path.
#
# This helper defaults to the native /embedding contract because the
# OpenAI-compatible /v1/embeddings route cannot carry raw token-sequence
# embeddings when the upstream backend runs with pooling disabled.

NATIVE_EMBEDDING_URL="${QWEN_NATIVE_EMBEDDING_URL:-http://127.0.0.1:8080/embedding}"
DEFAULT_PROXY_URL="${QWEN_PROXY_URL:-${PROXY_URL:-https://gpu.adriens-apis.io/llm/v1/embeddings}}"
if [[ "$DEFAULT_PROXY_URL" == */v1/embeddings ]]; then
    DEFAULT_PROXY_URL="$NATIVE_EMBEDDING_URL"
fi

export QWEN_OUTPUT_DIR="${QWEN_OUTPUT_DIR:-embedded_chunks/qwen3.5-27b}"
export QWEN_PROXY_URL="${QWEN_PROXY_URL:-$DEFAULT_PROXY_URL}"
export QWEN_PROXY_API_STYLE="${QWEN_PROXY_API_STYLE:-native_single}"
export QWEN_EXPECT_EMBEDDING_FORMAT="${QWEN_EXPECT_EMBEDDING_FORMAT:-token_sequence}"
export QWEN_EMBEDDING_POOLING="${QWEN_EMBEDDING_POOLING:-}"
export QWEN_EMBEDDING_EXTRA_BODY="${QWEN_EMBEDDING_EXTRA_BODY:-}"
export QWEN_SINGLE_REQUEST_CONCURRENCY="${QWEN_SINGLE_REQUEST_CONCURRENCY:-8}"
export QWEN_SINGLE_REQUEST_RETRIES="${QWEN_SINGLE_REQUEST_RETRIES:-3}"
export QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-128}"
export QWEN_ARCHIVE_EVERY="${QWEN_ARCHIVE_EVERY:-$((50 * QWEN_BATCH_SIZE))}"
export QWEN_CHECKPOINT_EVERY="${QWEN_CHECKPOINT_EVERY:-$((25 * QWEN_BATCH_SIZE))}"

echo "Starting Qwen no-pooling extraction"
echo "Qwen output root: $QWEN_OUTPUT_DIR"
echo "Qwen proxy URL: $QWEN_PROXY_URL"
echo "Qwen proxy API style: $QWEN_PROXY_API_STYLE"
echo "Qwen expected format: $QWEN_EXPECT_EMBEDDING_FORMAT"
echo "Qwen archive rows: $QWEN_ARCHIVE_EVERY"
echo "Qwen checkpoint rows: $QWEN_CHECKPOINT_EVERY"
echo "Qwen native request limits: concurrency=$QWEN_SINGLE_REQUEST_CONCURRENCY retries=$QWEN_SINGLE_REQUEST_RETRIES"
echo "Qwen request pooling: ${QWEN_EMBEDDING_POOLING:-default}"
if [[ -n "$QWEN_EMBEDDING_EXTRA_BODY" ]]; then
    echo "Qwen extra request body: $QWEN_EMBEDDING_EXTRA_BODY"
fi

exec env \
    QWEN_OUTPUT_DIR="$QWEN_OUTPUT_DIR" \
    QWEN_PROXY_URL="$QWEN_PROXY_URL" \
    QWEN_PROXY_API_STYLE="$QWEN_PROXY_API_STYLE" \
    QWEN_EXPECT_EMBEDDING_FORMAT="$QWEN_EXPECT_EMBEDDING_FORMAT" \
    QWEN_EMBEDDING_POOLING="$QWEN_EMBEDDING_POOLING" \
    QWEN_EMBEDDING_EXTRA_BODY="$QWEN_EMBEDDING_EXTRA_BODY" \
    QWEN_SINGLE_REQUEST_CONCURRENCY="$QWEN_SINGLE_REQUEST_CONCURRENCY" \
    QWEN_SINGLE_REQUEST_RETRIES="$QWEN_SINGLE_REQUEST_RETRIES" \
    QWEN_BATCH_SIZE="$QWEN_BATCH_SIZE" \
    QWEN_ARCHIVE_EVERY="$QWEN_ARCHIVE_EVERY" \
    QWEN_CHECKPOINT_EVERY="$QWEN_CHECKPOINT_EVERY" \
    uv run python runner.py