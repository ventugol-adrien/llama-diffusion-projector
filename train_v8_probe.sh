#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Probe archive shard sizes for v8 in-memory resampler training.
#
# The script builds temporary re-sharded Qwen and SDXL target roots, runs short
# v8 memory-mode probes, and writes ranked throughput summaries.
#
# Defaults test matched Qwen/CLIP row counts across a conservative max-row sweep.
# Set PAIR_MODE=matrix to test every QWEN_SHARD_ROWS_LIST x CLIP_SHARD_ROWS_LIST
# combination, or set PAIRS as space-separated qwen:clip pairs, for example:
#   PAIRS="12800:25600 25600:25600" ./train_v8_probe.sh

SOURCE_QWEN_OUTPUT_DIR="${SOURCE_QWEN_OUTPUT_DIR:-${QWEN_OUTPUT_DIR:-embedded_chunks/qwen3.5-27b}}"
SOURCE_QWEN_ARCHIVE_DIR="${SOURCE_QWEN_ARCHIVE_DIR:-$SOURCE_QWEN_OUTPUT_DIR/archive}"
SOURCE_QWEN_MANIFEST="${SOURCE_QWEN_MANIFEST:-$SOURCE_QWEN_OUTPUT_DIR/source_manifest.json}"
SOURCE_CLIP_OUTPUT_DIR="${SOURCE_CLIP_OUTPUT_DIR:-${TARGET_OUTPUT_DIR:-embedded_chunks/clip_archive}}"
if [[ -n "${SOURCE_CLIP_ARCHIVE_DIR:-}" ]]; then
    SOURCE_CLIP_ARCHIVE_DIR="$SOURCE_CLIP_ARCHIVE_DIR"
elif compgen -G "$SOURCE_CLIP_OUTPUT_DIR/target_archive_*.npz" >/dev/null; then
    SOURCE_CLIP_ARCHIVE_DIR="$SOURCE_CLIP_OUTPUT_DIR"
else
    SOURCE_CLIP_ARCHIVE_DIR="$SOURCE_CLIP_OUTPUT_DIR/targets/sdxl/archive"
fi

PROBE_ROOT="${PROBE_ROOT:-embedded_chunks/v8_archive_probe_$(date +%Y%m%d_%H%M%S)}"
PROBE_MAX_ROWS="${PROBE_MAX_ROWS:-}"
PROBE_MAX_ROWS_LIST="${PROBE_MAX_ROWS_LIST:-25000 50000 75000}"
ALLOW_FULL_DATASET_PROBE="${ALLOW_FULL_DATASET_PROBE:-0}"
SHARD_ROWS_LIST="${SHARD_ROWS_LIST:-6400 12800 25600 51200 65536}"
QWEN_SHARD_ROWS_LIST="${QWEN_SHARD_ROWS_LIST:-$SHARD_ROWS_LIST}"
CLIP_SHARD_ROWS_LIST="${CLIP_SHARD_ROWS_LIST:-$SHARD_ROWS_LIST}"
PAIR_MODE="${PAIR_MODE:-matched}"
PAIRS="${PAIRS:-}"

PROBE_TRAIN_STEPS="${PROBE_TRAIN_STEPS:-300}"
PROBE_TRAIN_BATCH_SIZE="${PROBE_TRAIN_BATCH_SIZE:-512}"
PROBE_PROGRESS_EVERY="${PROBE_PROGRESS_EVERY:-10}"
PROBE_VAL_SPLIT="${PROBE_VAL_SPLIT:-0}"
PROBE_CPU_THREADS="${PROBE_CPU_THREADS:-${CPU_THREADS:-7}}"
PROBE_NUM_WORKERS="${PROBE_NUM_WORKERS:-0}"
PROBE_STANDARDIZE="${PROBE_STANDARDIZE:-0}"
PROBE_CACHE_STANDARDIZED="${PROBE_CACHE_STANDARDIZED:-0}"
PROBE_MEMORY_MAX_GB="${PROBE_MEMORY_MAX_GB:-80}"
PROBE_USE_SYSTEMD_SCOPE="${PROBE_USE_SYSTEMD_SCOPE:-1}"
PROBE_MONITOR_INTERVAL="${PROBE_MONITOR_INTERVAL:-2}"

RESHARD_COMPRESS="${RESHARD_COMPRESS:-0}"
FORCE_RESHARD="${FORCE_RESHARD:-0}"
KEEP_RESHARDS="${KEEP_RESHARDS:-0}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
PROBE_DRY_RUN="${PROBE_DRY_RUN:-0}"
PROBE_RESHARD_ONLY="${PROBE_RESHARD_ONLY:-0}"

SUMMARY_CSV="$PROBE_ROOT/summary.csv"
DETAILS_JSONL="$PROBE_ROOT/details.jsonl"
EVENTS_JSONL="$PROBE_ROOT/events.jsonl"

mkdir -p "$PROBE_ROOT"

if [[ ! -d "$SOURCE_QWEN_ARCHIVE_DIR" ]]; then
    echo "Missing Qwen source archive dir: $SOURCE_QWEN_ARCHIVE_DIR" >&2
    exit 1
fi
if [[ ! -d "$SOURCE_CLIP_ARCHIVE_DIR" ]]; then
    echo "Missing CLIP source archive dir: $SOURCE_CLIP_ARCHIVE_DIR" >&2
    exit 1
fi
if [[ ! -f "$SOURCE_QWEN_MANIFEST" ]]; then
    echo "Missing Qwen source manifest: $SOURCE_QWEN_MANIFEST" >&2
    exit 1
fi

build_pairs() {
    if [[ -n "$PAIRS" ]]; then
        printf '%s\n' $PAIRS
        return
    fi

    case "$PAIR_MODE" in
        matched)
            for rows in $SHARD_ROWS_LIST; do
                printf '%s:%s\n' "$rows" "$rows"
            done
            ;;
        matrix)
            for qwen_rows in $QWEN_SHARD_ROWS_LIST; do
                for clip_rows in $CLIP_SHARD_ROWS_LIST; do
                    printf '%s:%s\n' "$qwen_rows" "$clip_rows"
                done
            done
            ;;
        *)
            echo "PAIR_MODE must be matched or matrix; got $PAIR_MODE" >&2
            exit 1
            ;;
    esac
}

echo "v8 archive-size probe"
echo "Source Qwen archives: $SOURCE_QWEN_ARCHIVE_DIR"
echo "Source CLIP archives: $SOURCE_CLIP_ARCHIVE_DIR"
echo "Probe root: $PROBE_ROOT"
if [[ -n "$PROBE_MAX_ROWS" ]]; then
    PROBE_MAX_ROWS_LIST="$PROBE_MAX_ROWS"
fi
for max_rows in $PROBE_MAX_ROWS_LIST; do
    if [[ ! "$max_rows" =~ ^[0-9]+$ ]]; then
        echo "Invalid max rows value '$max_rows'." >&2
        exit 1
    fi
    if [[ "$max_rows" == "0" && "$ALLOW_FULL_DATASET_PROBE" != "1" ]]; then
        echo "Refusing PROBE_MAX_ROWS=0 without ALLOW_FULL_DATASET_PROBE=1." >&2
        exit 1
    fi
done
echo "Probe max rows list: $PROBE_MAX_ROWS_LIST (0 means all rows, requires ALLOW_FULL_DATASET_PROBE=1)"
echo "Pair mode: $PAIR_MODE"
echo "Train steps: $PROBE_TRAIN_STEPS | batch size: $PROBE_TRAIN_BATCH_SIZE"
echo "Keep re-shards: $KEEP_RESHARDS | force re-shard: $FORCE_RESHARD"
echo "Memory limit: ${PROBE_MEMORY_MAX_GB}G via systemd scope: $PROBE_USE_SYSTEMD_SCOPE"

mapfile -t PROBE_PAIRS < <(build_pairs)
printf 'Pairs:\n'
printf '  %s\n' "${PROBE_PAIRS[@]}"

if [[ "$PROBE_DRY_RUN" == "1" ]]; then
    echo "Dry run only; exiting before re-shard/training."
    exit 0
fi

if [[ ! -f "$SUMMARY_CSV" ]]; then
    echo "max_rows,qwen_shard_rows,clip_shard_rows,status,exit_code,train_rate_last,train_rate_avg,train_rate_max,wall_seconds,max_rss_kb,gpu_util_avg,gpu_util_max,gpu_mem_max_mib,probe_log,time_log,gpu_log,resource_log,event_log,qwen_root,clip_root" > "$SUMMARY_CSV"
fi
: > "$DETAILS_JSONL"
: > "$EVENTS_JSONL"

log_event() {
    local event_log="$1"
    local candidate="$2"
    local stage="$3"
    local status="$4"
    local message="${5:-}"

    python3 - "$event_log" "$EVENTS_JSONL" "$candidate" "$stage" "$status" "$message" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

event_log, root_log, candidate, stage, status, message = sys.argv[1:]
payload = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "candidate": candidate,
    "stage": stage,
    "status": status,
    "message": message,
}
for path in (event_log, root_log):
    if not path:
        continue
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
PY
}

start_resource_monitor() {
    local resource_log="$1"
    local candidate="$2"
    local interval="$PROBE_MONITOR_INTERVAL"

    python3 - "$resource_log" "$candidate" "$interval" >/dev/null 2>&1 <<'PY' &
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

path, candidate, interval = sys.argv[1], sys.argv[2], float(sys.argv[3])
os.makedirs(os.path.dirname(path), exist_ok=True)

def read_meminfo():
    result = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            result[key] = int(value.strip().split()[0])
    return result

def read_gpuinfo():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip().splitlines()
    except Exception:
        return ["", "", "", "", ""]
    if not output:
        return ["", "", "", "", ""]
    return [item.strip() for item in output[0].split(",")]

fieldnames = [
    "ts",
    "candidate",
    "mem_total_kb",
    "mem_available_kb",
    "mem_free_kb",
    "swap_total_kb",
    "swap_free_kb",
    "gpu_util_pct",
    "gpu_mem_util_pct",
    "gpu_mem_used_mib",
    "gpu_mem_total_mib",
    "gpu_power_w",
]
with open(path, "a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if handle.tell() == 0:
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())
    while True:
        mem = read_meminfo()
        gpu = read_gpuinfo()
        writer.writerow(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "candidate": candidate,
                "mem_total_kb": mem.get("MemTotal", ""),
                "mem_available_kb": mem.get("MemAvailable", ""),
                "mem_free_kb": mem.get("MemFree", ""),
                "swap_total_kb": mem.get("SwapTotal", ""),
                "swap_free_kb": mem.get("SwapFree", ""),
                "gpu_util_pct": gpu[0],
                "gpu_mem_util_pct": gpu[1],
                "gpu_mem_used_mib": gpu[2],
                "gpu_mem_total_mib": gpu[3],
                "gpu_power_w": gpu[4],
            }
        )
        handle.flush()
        os.fsync(handle.fileno())
        time.sleep(interval)
PY
    echo "$!"
}

reshard_pair() {
    local qwen_rows="$1"
    local clip_rows="$2"
    local max_rows="$3"
    local qwen_root="$4"
    local clip_root="$5"

    if [[ "$FORCE_RESHARD" == "1" ]]; then
        rm -rf "$qwen_root" "$clip_root"
    fi

    if [[ -f "$qwen_root/.complete" && -f "$clip_root/.complete" ]]; then
        echo "Reusing existing re-shards: qwen=$qwen_root clip=$clip_root"
        return
    fi

    rm -rf "$qwen_root" "$clip_root"
    mkdir -p "$qwen_root" "$clip_root"

    uv run python - \
        --source-qwen-archive-dir "$SOURCE_QWEN_ARCHIVE_DIR" \
        --source-qwen-manifest "$SOURCE_QWEN_MANIFEST" \
        --source-clip-archive-dir "$SOURCE_CLIP_ARCHIVE_DIR" \
        --qwen-output-root "$qwen_root" \
        --clip-output-dir "$clip_root" \
        --qwen-shard-rows "$qwen_rows" \
        --clip-shard-rows "$clip_rows" \
        --max-rows "$max_rows" \
        --compress "$RESHARD_COMPRESS" <<'PY'
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-qwen-archive-dir", required=True)
    parser.add_argument("--source-qwen-manifest", required=True)
    parser.add_argument("--source-clip-archive-dir", required=True)
    parser.add_argument("--qwen-output-root", required=True)
    parser.add_argument("--clip-output-dir", required=True)
    parser.add_argument("--qwen-shard-rows", type=int, required=True)
    parser.add_argument("--clip-shard-rows", type=int, required=True)
    parser.add_argument("--max-rows", type=int, required=True)
    parser.add_argument("--compress", choices=("0", "1"), default="0")
    return parser.parse_args()


def list_archives(path: Path, prefix: str) -> list[Path]:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{7}})_(\d{{7}})\.npz$")
    archives = [child for child in path.iterdir() if child.is_file() and pattern.match(child.name)]
    archives.sort(key=lambda child: int(pattern.match(child.name).group(1)))
    if not archives:
        raise FileNotFoundError(f"No {prefix}_*.npz files found in {path}")
    return archives


def write_npz(path: Path, payload: dict[str, np.ndarray], compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as handle:
        if compress:
            np.savez_compressed(handle, **payload)
        else:
            np.savez(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def copy_manifest(source: Path, output_root: Path, shard_rows: int, max_rows: int) -> None:
    with open(source, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["archive_every"] = int(shard_rows)
    payload["probe_max_rows"] = int(max_rows)
    tmp = output_root / "source_manifest.json.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, output_root / "source_manifest.json")


class QwenWriter:
    def __init__(self, output_archive_dir: Path, shard_rows: int, compress: bool):
        self.output_archive_dir = output_archive_dir
        self.shard_rows = int(shard_rows)
        self.compress = bool(compress)
        self.indices_parts = []
        self.token_parts = []
        self.length_parts = []
        self.embedding_parts = []
        self.rows = 0
        self.total_rows = 0
        self.mode = None

    def add_vector(self, indices: np.ndarray, embeddings: np.ndarray) -> None:
        self._set_mode("vector")
        start = 0
        while start < len(indices):
            take = min(self.shard_rows - self.rows, len(indices) - start)
            end = start + take
            self.indices_parts.append(indices[start:end].copy())
            self.embedding_parts.append(embeddings[start:end].copy())
            self.rows += take
            self.total_rows += take
            if self.rows == self.shard_rows:
                self.flush()
            start = end

    def add_tokens(self, indices: np.ndarray, token_embeddings: np.ndarray, token_offsets: np.ndarray, sequence_lengths: np.ndarray) -> None:
        self._set_mode("token_sequence")
        start = 0
        while start < len(indices):
            take = min(self.shard_rows - self.rows, len(indices) - start)
            end = start + take
            token_start = int(token_offsets[start])
            token_end = int(token_offsets[end])
            self.indices_parts.append(indices[start:end].copy())
            self.token_parts.append(token_embeddings[token_start:token_end].copy())
            self.length_parts.append(sequence_lengths[start:end].copy())
            self.rows += take
            self.total_rows += take
            if self.rows == self.shard_rows:
                self.flush()
            start = end

    def _set_mode(self, mode: str) -> None:
        if self.mode is None:
            self.mode = mode
        elif self.mode != mode:
            raise RuntimeError("Mixed Qwen archive formats are not supported")

    def flush(self) -> None:
        if self.rows == 0:
            return
        indices = np.concatenate(self.indices_parts, axis=0)
        first = int(indices[0])
        last = int(indices[-1])
        payload = {"indices": indices}
        if self.mode == "token_sequence":
            lengths = np.concatenate(self.length_parts, axis=0).astype(np.int32, copy=False)
            offsets = np.empty(len(lengths) + 1, dtype=np.int64)
            offsets[0] = 0
            np.cumsum(lengths.astype(np.int64, copy=False), out=offsets[1:])
            payload.update(
                {
                    "token_embeddings": np.concatenate(self.token_parts, axis=0).astype(np.float32, copy=False),
                    "token_offsets": offsets,
                    "sequence_lengths": lengths,
                }
            )
        else:
            payload["embeddings"] = np.concatenate(self.embedding_parts, axis=0).astype(np.float32, copy=False)
        output = self.output_archive_dir / f"archive_{first:07d}_{last:07d}.npz"
        write_npz(output, payload, self.compress)
        print(f"Wrote Qwen {self.rows:,} rows -> {output}")
        self.indices_parts.clear()
        self.token_parts.clear()
        self.length_parts.clear()
        self.embedding_parts.clear()
        self.rows = 0


class ClipWriter:
    def __init__(self, output_dir: Path, shard_rows: int, compress: bool):
        self.output_dir = output_dir
        self.shard_rows = int(shard_rows)
        self.compress = bool(compress)
        self.indices_parts = []
        self.prompt_parts = []
        self.pooled_parts = []
        self.rows = 0
        self.total_rows = 0

    def add(self, indices: np.ndarray, prompt: np.ndarray, pooled: np.ndarray) -> None:
        start = 0
        while start < len(indices):
            take = min(self.shard_rows - self.rows, len(indices) - start)
            end = start + take
            self.indices_parts.append(indices[start:end].copy())
            self.prompt_parts.append(prompt[start:end].copy())
            self.pooled_parts.append(pooled[start:end].copy())
            self.rows += take
            self.total_rows += take
            if self.rows == self.shard_rows:
                self.flush()
            start = end

    def flush(self) -> None:
        if self.rows == 0:
            return
        indices = np.concatenate(self.indices_parts, axis=0)
        first = int(indices[0])
        last = int(indices[-1])
        payload = {
            "indices": indices,
            "prompt_embeds": np.concatenate(self.prompt_parts, axis=0),
            "pooled_prompt_embeds": np.concatenate(self.pooled_parts, axis=0),
        }
        output = self.output_dir / f"target_archive_{first:07d}_{last:07d}.npz"
        write_npz(output, payload, self.compress)
        print(f"Wrote CLIP {self.rows:,} rows -> {output}")
        self.indices_parts.clear()
        self.prompt_parts.clear()
        self.pooled_parts.clear()
        self.rows = 0


def reshard_qwen(source_dir: Path, output_root: Path, manifest: Path, shard_rows: int, max_rows: int, compress: bool) -> int:
    output_archive_dir = output_root / "archive"
    output_archive_dir.mkdir(parents=True, exist_ok=True)
    writer = QwenWriter(output_archive_dir, shard_rows, compress)
    rows_left = max_rows if max_rows > 0 else None
    for archive in list_archives(source_dir, "archive"):
        if rows_left == 0:
            break
        with np.load(archive, allow_pickle=False) as data:
            take = len(data["indices"]) if rows_left is None else min(rows_left, len(data["indices"]))
            indices = data["indices"][:take].astype(np.int64, copy=False)
            if "token_embeddings" in data and "token_offsets" in data:
                offsets = data["token_offsets"].astype(np.int64, copy=False)
                token_end = int(offsets[take])
                lengths = (
                    data["sequence_lengths"][:take].astype(np.int32, copy=False)
                    if "sequence_lengths" in data
                    else np.diff(offsets[: take + 1]).astype(np.int32, copy=False)
                )
                writer.add_tokens(indices, data["token_embeddings"][:token_end], offsets[: take + 1], lengths)
            elif "embeddings" in data:
                writer.add_vector(indices, data["embeddings"][:take])
            else:
                raise RuntimeError(f"Unsupported Qwen archive payload: {archive}")
        if rows_left is not None:
            rows_left -= take
    writer.flush()
    copy_manifest(manifest, output_root, shard_rows, max_rows)
    return writer.total_rows


def reshard_clip(source_dir: Path, output_dir: Path, shard_rows: int, max_rows: int, compress: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = ClipWriter(output_dir, shard_rows, compress)
    rows_left = max_rows if max_rows > 0 else None
    for archive in list_archives(source_dir, "target_archive"):
        if rows_left == 0:
            break
        with np.load(archive, allow_pickle=False) as data:
            take = len(data["indices"]) if rows_left is None else min(rows_left, len(data["indices"]))
            writer.add(
                data["indices"][:take].astype(np.int64, copy=False),
                data["prompt_embeds"][:take],
                data["pooled_prompt_embeds"][:take],
            )
        if rows_left is not None:
            rows_left -= take
    writer.flush()
    return writer.total_rows


args = parse_args()
qwen_root = Path(args.qwen_output_root)
clip_root = Path(args.clip_output_dir)
compress = args.compress == "1"
qwen_rows = reshard_qwen(Path(args.source_qwen_archive_dir), qwen_root, Path(args.source_qwen_manifest), args.qwen_shard_rows, args.max_rows, compress)
clip_rows = reshard_clip(Path(args.source_clip_archive_dir), clip_root, args.clip_shard_rows, args.max_rows, compress)
if qwen_rows != clip_rows:
    raise RuntimeError(f"Row mismatch after re-shard: qwen={qwen_rows:,}, clip={clip_rows:,}")
metadata = {
    "qwen_rows": qwen_rows,
    "clip_rows": clip_rows,
    "qwen_shard_rows": args.qwen_shard_rows,
    "clip_shard_rows": args.clip_shard_rows,
    "max_rows": args.max_rows,
}
for root in (qwen_root, clip_root):
    tmp = root / ".complete.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, root / ".complete")
print(json.dumps(metadata, indent=2, sort_keys=True))
PY
}

append_result() {
    local max_rows="$1"
    local qwen_rows="$2"
    local clip_rows="$3"
    local status="$4"
    local exit_code="$5"
    local probe_log="$6"
    local time_log="$7"
    local gpu_log="$8"
    local resource_log="$9"
    local event_log="${10}"
    local qwen_root="${11}"
    local clip_root="${12}"

    uv run python - \
        --summary "$SUMMARY_CSV" \
        --details "$DETAILS_JSONL" \
        --max-rows "$max_rows" \
        --qwen-rows "$qwen_rows" \
        --clip-rows "$clip_rows" \
        --status "$status" \
        --exit-code "$exit_code" \
        --probe-log "$probe_log" \
        --time-log "$time_log" \
        --gpu-log "$gpu_log" \
        --resource-log "$resource_log" \
        --event-log "$event_log" \
        --qwen-root "$qwen_root" \
        --clip-root "$clip_root" <<'PY'
import argparse
import csv
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--details", required=True)
    parser.add_argument("--max-rows", required=True)
    parser.add_argument("--qwen-rows", required=True)
    parser.add_argument("--clip-rows", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--exit-code", required=True)
    parser.add_argument("--probe-log", required=True)
    parser.add_argument("--time-log", required=True)
    parser.add_argument("--gpu-log", required=True)
    parser.add_argument("--resource-log", required=True)
    parser.add_argument("--event-log", required=True)
    parser.add_argument("--qwen-root", required=True)
    parser.add_argument("--clip-root", required=True)
    return parser.parse_args()


def elapsed_to_seconds(text: str):
    parts = text.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except Exception:
        return ""


def parse_time(path: Path):
    result = {"wall_seconds": "", "max_rss_kb": ""}
    if not path.exists():
        return result
    text = path.read_text(errors="replace")
    elapsed = re.search(r"Elapsed \(wall clock\) time .*: ([^\n]+)", text)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    if elapsed:
        result["wall_seconds"] = elapsed_to_seconds(elapsed.group(1))
    if rss:
        result["max_rss_kb"] = int(rss.group(1))
    return result


def parse_rates(path: Path):
    if not path.exists():
        return {"last": "", "avg": "", "max": "", "rates": []}
    text = path.read_text(errors="replace")
    rates = [float(value) for value in re.findall(r"rate=([0-9]+(?:\.[0-9]+)?) rows/s", text)]
    if not rates:
        return {"last": "", "avg": "", "max": "", "rates": []}
    return {
        "last": rates[-1],
        "avg": sum(rates) / len(rates),
        "max": max(rates),
        "rates": rates,
    }


def parse_gpu(path: Path):
    result = {"gpu_util_avg": "", "gpu_util_max": "", "gpu_mem_max_mib": ""}
    if not path.exists() or path.stat().st_size == 0:
        return result
    utils = []
    mems = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            util = row.get(" utilization.gpu [%]") or row.get("utilization.gpu [%]")
            mem = row.get(" memory.used [MiB]") or row.get("memory.used [MiB]")
            if util:
                utils.append(float(util.replace("%", "").strip()))
            if mem:
                mems.append(float(mem.replace("MiB", "").strip()))
    if utils:
        result["gpu_util_avg"] = sum(utils) / len(utils)
        result["gpu_util_max"] = max(utils)
    if mems:
        result["gpu_mem_max_mib"] = max(mems)
    return result


args = parse_args()
rates = parse_rates(Path(args.probe_log))
timing = parse_time(Path(args.time_log))
gpu = parse_gpu(Path(args.gpu_log))
row = {
    "max_rows": args.max_rows,
    "qwen_shard_rows": args.qwen_rows,
    "clip_shard_rows": args.clip_rows,
    "status": args.status,
    "exit_code": args.exit_code,
    "train_rate_last": rates["last"],
    "train_rate_avg": rates["avg"],
    "train_rate_max": rates["max"],
    "wall_seconds": timing["wall_seconds"],
    "max_rss_kb": timing["max_rss_kb"],
    "gpu_util_avg": gpu["gpu_util_avg"],
    "gpu_util_max": gpu["gpu_util_max"],
    "gpu_mem_max_mib": gpu["gpu_mem_max_mib"],
    "probe_log": args.probe_log,
    "time_log": args.time_log,
    "gpu_log": args.gpu_log,
    "resource_log": args.resource_log,
    "event_log": args.event_log,
    "qwen_root": args.qwen_root,
    "clip_root": args.clip_root,
}
with open(args.summary, "a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(row))
    writer.writerow(row)
with open(args.details, "a", encoding="utf-8") as handle:
    detail = dict(row)
    detail["rates"] = rates["rates"]
    handle.write(json.dumps(detail, sort_keys=True) + "\n")
PY
}

rank_results() {
    uv run python - "$SUMMARY_CSV" <<'PY'
import csv
import sys
from pathlib import Path

summary = Path(sys.argv[1])
if not summary.exists():
    raise SystemExit(0)
with open(summary, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

def score(row):
    try:
        return float(row.get("train_rate_avg") or 0.0)
    except ValueError:
        return 0.0

rows.sort(key=score, reverse=True)
print("\nRanked probe results by average train rows/s:")
for idx, row in enumerate(rows, 1):
    print(
        f"{idx:2d}. max_rows={row.get('max_rows', 'n/a'):>6} qwen={row['qwen_shard_rows']:>6} clip={row['clip_shard_rows']:>6} "
        f"status={row['status']:<9} avg={row['train_rate_avg'] or 'n/a'} "
        f"last={row['train_rate_last'] or 'n/a'} max={row['train_rate_max'] or 'n/a'} "
        f"rss_kb={row['max_rss_kb'] or 'n/a'} gpu_avg={row['gpu_util_avg'] or 'n/a'}"
    )
PY
}

for max_rows in $PROBE_MAX_ROWS_LIST; do
for pair in "${PROBE_PAIRS[@]}"; do
    qwen_rows="${pair%%:*}"
    clip_rows="${pair##*:}"
    if [[ ! "$qwen_rows" =~ ^[0-9]+$ || ! "$clip_rows" =~ ^[0-9]+$ ]]; then
        echo "Invalid pair '$pair'; expected qwen_rows:clip_rows" >&2
        exit 1
    fi

    suffix="rows${max_rows}"
    if [[ "$max_rows" == "0" ]]; then
        suffix="all"
    fi
    candidate="q${qwen_rows}_c${clip_rows}_${suffix}"
    qwen_root="$PROBE_ROOT/reshards/$candidate/qwen"
    clip_root="$PROBE_ROOT/reshards/$candidate/clip"
    run_dir="$PROBE_ROOT/runs/$candidate"
    mkdir -p "$run_dir"

    printf '\n=== Candidate: max_rows=%s qwen_rows=%s clip_rows=%s ===\n' "$max_rows" "$qwen_rows" "$clip_rows"
    event_log="$run_dir/events.jsonl"
    resource_log="$run_dir/resources.csv"
    log_event "$event_log" "$candidate" "candidate" "start" "max_rows=$max_rows qwen_rows=$qwen_rows clip_rows=$clip_rows"
    log_event "$event_log" "$candidate" "reshard" "start" "qwen_root=$qwen_root clip_root=$clip_root"
    reshard_pair "$qwen_rows" "$clip_rows" "$max_rows" "$qwen_root" "$clip_root"
    log_event "$event_log" "$candidate" "reshard" "complete" "re-shard complete"

    if [[ "$PROBE_RESHARD_ONLY" == "1" ]]; then
        append_result "$max_rows" "$qwen_rows" "$clip_rows" "resharded" "0" "/dev/null" "/dev/null" "/dev/null" "$resource_log" "$event_log" "$qwen_root" "$clip_root"
        continue
    fi

    probe_log="$run_dir/probe.log"
    time_log="$run_dir/time.log"
    gpu_log="$run_dir/gpu.csv"
    gpu_sampler_pid=""
    resource_sampler_pid="$(start_resource_monitor "$resource_log" "$candidate")"
    log_event "$event_log" "$candidate" "train" "start" "starting memory-mode v8 probe"

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi \
            --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
            --format=csv \
            -l 1 > "$gpu_log" &
        gpu_sampler_pid="$!"
    else
        : > "$gpu_log"
    fi

    set +e
    if [[ "$PROBE_USE_SYSTEMD_SCOPE" == "1" ]] && command -v systemd-run >/dev/null 2>&1; then
        systemd-run --scope -q \
            -p "MemoryMax=${PROBE_MEMORY_MAX_GB}G" \
            -p "MemorySwapMax=0" \
            /usr/bin/time -v -o "$time_log" \
            env \
                QWEN_OUTPUT_DIR="$qwen_root" \
                TARGET_OUTPUT_DIR="$clip_root" \
                TRAIN_ARCHIVE_MODE="memory" \
                TRAIN_TARGET_FAMILY="sdxl" \
                TRAIN_SDXL_PROJECTOR_ARCH="resampler" \
                TRAIN_MAX_STEPS="$PROBE_TRAIN_STEPS" \
                TRAIN_MAX_SAMPLES="$max_rows" \
                TRAIN_BATCH_SIZE="$PROBE_TRAIN_BATCH_SIZE" \
                TRAIN_VAL_SPLIT="$PROBE_VAL_SPLIT" \
                TRAIN_EPOCHS="1" \
                TRAIN_PROGRESS_EVERY="$PROBE_PROGRESS_EVERY" \
                TRAIN_PROGRESS_SECONDS="0" \
                TRAIN_TIMING="0" \
                TRAIN_AUTO_RESUME="0" \
                TRAIN_STANDARDIZE_EMBEDDINGS="$PROBE_STANDARDIZE" \
                TRAIN_CACHE_STANDARDIZED_ARCHIVES="$PROBE_CACHE_STANDARDIZED" \
                TRAIN_NUM_WORKERS="$PROBE_NUM_WORKERS" \
                CPU_THREADS="$PROBE_CPU_THREADS" \
                RUN_LABEL="probe_${candidate}" \
                TRAIN_BEST_CHECKPOINT="$run_dir/best.pt" \
                TRAIN_RESUME_CHECKPOINT="$run_dir/resume.pt" \
                TRAIN_GGUF_PATH="$run_dir/probe.gguf" \
                "$ROOT_DIR/train_v8.sh" 2>&1 | tee "$probe_log"
        exit_code="${PIPESTATUS[0]}"
    else
        /usr/bin/time -v -o "$time_log" \
            env \
                QWEN_OUTPUT_DIR="$qwen_root" \
                TARGET_OUTPUT_DIR="$clip_root" \
                TRAIN_ARCHIVE_MODE="memory" \
                TRAIN_TARGET_FAMILY="sdxl" \
                TRAIN_SDXL_PROJECTOR_ARCH="resampler" \
                TRAIN_MAX_STEPS="$PROBE_TRAIN_STEPS" \
                TRAIN_MAX_SAMPLES="$max_rows" \
                TRAIN_BATCH_SIZE="$PROBE_TRAIN_BATCH_SIZE" \
                TRAIN_VAL_SPLIT="$PROBE_VAL_SPLIT" \
                TRAIN_EPOCHS="1" \
                TRAIN_PROGRESS_EVERY="$PROBE_PROGRESS_EVERY" \
                TRAIN_PROGRESS_SECONDS="0" \
                TRAIN_TIMING="0" \
                TRAIN_AUTO_RESUME="0" \
                TRAIN_STANDARDIZE_EMBEDDINGS="$PROBE_STANDARDIZE" \
                TRAIN_CACHE_STANDARDIZED_ARCHIVES="$PROBE_CACHE_STANDARDIZED" \
                TRAIN_NUM_WORKERS="$PROBE_NUM_WORKERS" \
                CPU_THREADS="$PROBE_CPU_THREADS" \
                RUN_LABEL="probe_${candidate}" \
                TRAIN_BEST_CHECKPOINT="$run_dir/best.pt" \
                TRAIN_RESUME_CHECKPOINT="$run_dir/resume.pt" \
                TRAIN_GGUF_PATH="$run_dir/probe.gguf" \
                "$ROOT_DIR/train_v8.sh" 2>&1 | tee "$probe_log"
        exit_code="${PIPESTATUS[0]}"
    fi
    set -e

    if [[ -n "$resource_sampler_pid" ]]; then
        kill "$resource_sampler_pid" >/dev/null 2>&1 || true
        wait "$resource_sampler_pid" >/dev/null 2>&1 || true
    fi

    if [[ -n "$gpu_sampler_pid" ]]; then
        kill "$gpu_sampler_pid" >/dev/null 2>&1 || true
        wait "$gpu_sampler_pid" >/dev/null 2>&1 || true
    fi

    status="ok"
    if [[ "$exit_code" != "0" ]]; then
        status="failed"
    fi
    log_event "$event_log" "$candidate" "train" "$status" "exit_code=$exit_code"
    append_result "$max_rows" "$qwen_rows" "$clip_rows" "$status" "$exit_code" "$probe_log" "$time_log" "$gpu_log" "$resource_log" "$event_log" "$qwen_root" "$clip_root"

    if [[ "$KEEP_RESHARDS" != "1" ]]; then
        rm -rf "$PROBE_ROOT/reshards/$candidate"
    fi

    rank_results

    if [[ "$exit_code" != "0" && "$STOP_ON_FAILURE" == "1" ]]; then
        echo "Stopping after failed candidate $candidate" >&2
        exit "$exit_code"
    fi
done
done

rank_results
printf '\nSummary CSV: %s\n' "$SUMMARY_CSV"
printf 'Details JSONL: %s\n' "$DETAILS_JSONL"