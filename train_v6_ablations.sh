#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Short, runnable v6 ablations over the currently implemented surfaces.
#
# This ladder targets the next scale-remediation slice after standardized-space
# input/output correction:
# - standardized baseline
# - standardized + spectral norm
# - standardized + calibrator
# - standardized + spectral norm + calibrator

TARGET_OUTPUT_DIR="${TARGET_OUTPUT_DIR:-embedded_chunks/run-1m-2560}"
QWEN_OUTPUT_DIR="${QWEN_OUTPUT_DIR:-embedded_chunks}"
CPU_THREADS="${CPU_THREADS:-8}"
TRAIN_ARCHIVE_THREADS="${TRAIN_ARCHIVE_THREADS:-8}"
TRAIN_STANDARDIZATION_THREADS="${TRAIN_STANDARDIZATION_THREADS:-$CPU_THREADS}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
ABLATION_EPOCHS="${ABLATION_EPOCHS:-8}"
ABLATION_MAX_STEPS="${ABLATION_MAX_STEPS:-1500}"
ABLATION_VARIANTS="${ABLATION_VARIANTS:-baseline spectral_norm calibrator spectral_norm_calibrator}"

run_variant() {
    local variant="$1"
    local run_label=""
    local spectral_norm="0"
    local calibrator="0"

    case "$variant" in
        baseline)
            run_label="v6a_std_base"
            ;;
        spectral_norm)
            run_label="v6b_std_spectral"
            spectral_norm="1"
            ;;
        calibrator)
            run_label="v6c_std_calibrator"
            calibrator="1"
            ;;
        spectral_norm_calibrator)
            run_label="v6d_std_spectral_calibrator"
            spectral_norm="1"
            calibrator="1"
            ;;
        *)
            echo "Unknown ablation variant: $variant" >&2
            echo "Expected one of: baseline spectral_norm calibrator spectral_norm_calibrator" >&2
            return 1
            ;;
    esac

    echo "==> Running v6 ablation: $variant (label=$run_label)"
    env \
        TARGET_OUTPUT_DIR="$TARGET_OUTPUT_DIR" \
        QWEN_OUTPUT_DIR="$QWEN_OUTPUT_DIR" \
        CPU_THREADS="$CPU_THREADS" \
        TRAIN_ARCHIVE_THREADS="$TRAIN_ARCHIVE_THREADS" \
        TRAIN_STANDARDIZATION_THREADS="$TRAIN_STANDARDIZATION_THREADS" \
        TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
        TRAIN_EPOCHS="$ABLATION_EPOCHS" \
        TRAIN_MAX_STEPS="$ABLATION_MAX_STEPS" \
        RUN_LABEL="$run_label" \
        TRAIN_SDXL_SPECTRAL_NORM="$spectral_norm" \
        TRAIN_SDXL_USE_OUTPUT_CALIBRATOR="$calibrator" \
        "$ROOT_DIR/train_v6.sh"
}

for variant in $ABLATION_VARIANTS; do
    run_variant "$variant"
done