import os
import math
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

# RDNA4 (gfx1200) can require an explicit override on ROCm so PyTorch builds
# the correct kernels instead of falling back to slower generic behavior.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "12.0.0")

import gguf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, Subset

from target_families import ensure_target_root, get_target_family, get_target_layout


def _env_int(name, default):
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name, default):
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_str(name, default):
    value = os.getenv(name)
    return default if value is None else value


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean string (0/1, false/true, no/yes, off/on). "
        f"Got '{value}'."
    )


@dataclass(frozen=True)
class SDXLossConfig:
    pointwise_name: str = "mse"
    huber_delta: float = 1.0
    prompt_cosine_weight: float = 0.0
    pooled_cosine_weight: float = 0.0
    prompt_norm_weight: float = 0.0
    pooled_norm_weight: float = 0.0
    prompt_norm_match_weight: float = 0.0
    pooled_norm_match_weight: float = 0.0


@dataclass(frozen=True)
class SDXLMonitorConfig:
    metric_name: str = "mse"
    norm_ratio_weight: float = 0.0
    std_ratio_weight: float = 0.0


def _build_sdxl_loss_config():
    pointwise_name = _env_str("TRAIN_SDXL_POINTWISE_LOSS", "mse").strip().lower()
    if pointwise_name not in {"mse", "huber"}:
        raise ValueError("TRAIN_SDXL_POINTWISE_LOSS must be 'mse' or 'huber'.")
    return SDXLossConfig(
        pointwise_name=pointwise_name,
        huber_delta=max(_env_float("TRAIN_SDXL_HUBER_DELTA", 1.0), 1e-6),
        prompt_cosine_weight=_env_float("TRAIN_SDXL_PROMPT_COSINE_WEIGHT", 0.05),
        pooled_cosine_weight=_env_float("TRAIN_SDXL_POOLED_COSINE_WEIGHT", 0.05),
        prompt_norm_weight=_env_float("TRAIN_SDXL_PROMPT_NORM_WEIGHT", 0.05),
        pooled_norm_weight=_env_float("TRAIN_SDXL_POOLED_NORM_WEIGHT", 0.05),
        prompt_norm_match_weight=_env_float("TRAIN_SDXL_PROMPT_NORM_MATCH_WEIGHT", 0.0),
        pooled_norm_match_weight=_env_float("TRAIN_SDXL_POOLED_NORM_MATCH_WEIGHT", 0.0),
    )


def _build_sdxl_monitor_config():
    metric_name = _env_str("TRAIN_SDXL_MONITOR_METRIC", "mse").strip().lower()
    if metric_name not in {"mse", "composite"}:
        raise ValueError("TRAIN_SDXL_MONITOR_METRIC must be 'mse' or 'composite'.")
    return SDXLMonitorConfig(
        metric_name=metric_name,
        norm_ratio_weight=max(_env_float("TRAIN_SDXL_MONITOR_NORM_WEIGHT", 0.0), 0.0),
        std_ratio_weight=max(_env_float("TRAIN_SDXL_MONITOR_STD_WEIGHT", 0.0), 0.0),
    )


def _build_pointwise_criterion(target_family, sdxl_loss_config=None):
    if target_family == "sdxl" and sdxl_loss_config is not None:
        if sdxl_loss_config.pointwise_name == "huber":
            return nn.HuberLoss(delta=sdxl_loss_config.huber_delta)
    return nn.MSELoss()


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


def _should_emit_progress(step, last_emit_s, progress_every, progress_seconds):
    if progress_every > 0 and step % progress_every == 0:
        return True
    if progress_seconds > 0 and perf_counter() - last_emit_s >= progress_seconds:
        return True
    return False


def _bundle_first_tensor(bundle):
    return next(iter(bundle.values()))


def _bundle_batch_size(bundle):
    return int(_bundle_first_tensor(bundle).shape[0])


def _pin_target_bundle(bundle):
    return {key: value.pin_memory() for key, value in bundle.items()}


def _move_target_bundle_to_device(bundle, device, pin_memory):
    return {
        key: value.to(device, non_blocking=pin_memory) for key, value in bundle.items()
    }


def _select_target_bundle(bundle, indices):
    return {key: value.index_select(0, indices) for key, value in bundle.items()}


def _slice_target_bundle(bundle, start, end):
    return {key: value[start:end] for key, value in bundle.items()}


def _bundle_from_numpy(bundle):
    return {key: torch.from_numpy(value) for key, value in bundle.items()}


def _concat_target_bundles(*bundles):
    active_bundles = [bundle for bundle in bundles if bundle is not None]
    if not active_bundles:
        return {}
    if len(active_bundles) == 1:
        return active_bundles[0]
    return {
        key: torch.cat([bundle[key] for bundle in active_bundles], dim=0)
        for key in active_bundles[0]
    }


def _flatten_for_cosine(tensor):
    return tensor.reshape(tensor.shape[0], -1)


def _weighted_average(values, weights):
    total_weight = float(sum(weights))
    active_weights = list(weights)
    if total_weight <= 0:
        total_weight = float(len(values))
        active_weights = [1.0] * len(values)
    result = values[0] * 0
    for value, weight in zip(values, active_weights):
        result = result + value * weight
    return result / total_weight


def _init_metric_sums(target_family, device):
    metric_sums = {
        "mse": torch.zeros((), device=device),
        "mae": torch.zeros((), device=device),
        "cosine": torch.zeros((), device=device),
    }
    if target_family == "sdxl":
        metric_sums.update(
            {
                "prompt_mse": torch.zeros((), device=device),
                "prompt_mae": torch.zeros((), device=device),
                "prompt_cosine": torch.zeros((), device=device),
                "pooled_mse": torch.zeros((), device=device),
                "pooled_mae": torch.zeros((), device=device),
                "pooled_cosine": torch.zeros((), device=device),
            }
        )
    return metric_sums


def _accumulate_metric_sums(metric_sums, batch_metrics):
    for key, value in batch_metrics.items():
        if key not in metric_sums:
            metric_sums[key] = torch.zeros_like(value)
        metric_sums[key] += value


def _finalize_metric_sums(metric_sums, steps):
    return {key: (value / max(1, steps)).item() for key, value in metric_sums.items()}


def _start_timing_window(device, timing_enabled):
    if not timing_enabled:
        return None, None, None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return start_event, end_event, None
    return None, None, perf_counter()


def _finish_timing_window(start_event, end_event, start_s):
    if start_event is not None and end_event is not None:
        end_event.record()
        end_event.synchronize()
        return start_event.elapsed_time(end_event) / 1000.0
    if start_s is None:
        return 0.0
    return perf_counter() - start_s


def _finalize_stream_instrumentation(stats):
    if stats is None:
        return {}
    batch_count = max(1, stats["batch_count"])
    chunk_count = max(1, stats["chunk_count"])
    return {
        "avg_batch_rows": stats["batch_rows"] / batch_count,
        "avg_chunk_rows": stats["chunk_rows"] / chunk_count,
        "avg_chunk_wait_ms": (stats["chunk_wait_s"] * 1000.0) / chunk_count,
        "avg_copy_ms": (stats["copy_s"] * 1000.0) / batch_count,
        "avg_gpu_ms": (stats["compute_s"] * 1000.0) / batch_count,
    }


def _metrics_extra_text(metrics, prefix=""):
    parts = []
    if "monitor" in metrics and metrics.get("monitor_name", "mse") != "mse":
        parts.append(f" | {prefix}monitor={metrics['monitor']:.6f}")
    if "objective" in metrics and abs(metrics["objective"] - metrics["mse"]) > 1e-8:
        parts.append(f" | {prefix}obj={metrics['objective']:.6f}")
    if "pointwise" in metrics and abs(metrics["pointwise"] - metrics["mse"]) > 1e-8:
        parts.append(f" | {prefix}point={metrics['pointwise']:.6f}")
    if "prompt_mse" in metrics:
        parts.append(f" | {prefix}prompt_mse={metrics['prompt_mse']:.6f}")
        parts.append(f" | {prefix}pooled_mse={metrics['pooled_mse']:.6f}")
    if "prompt_norm_match_loss" in metrics:
        parts.append(f" | {prefix}prompt_dnorm={metrics['prompt_norm_match_loss']:.6f}")
        parts.append(f" | {prefix}pooled_dnorm={metrics['pooled_norm_match_loss']:.6f}")
    if "prompt_norm_loss" in metrics:
        parts.append(f" | {prefix}prompt_logn={metrics['prompt_norm_loss']:.6f}")
        parts.append(f" | {prefix}pooled_logn={metrics['pooled_norm_loss']:.6f}")
    if "scale_matched_mse" in metrics:
        parts.append(f" | {prefix}smse={metrics['scale_matched_mse']:.6f}")
    if "prompt_norm_ratio" in metrics:
        parts.append(f" | {prefix}prompt_nr={metrics['prompt_norm_ratio']:.3f}")
        parts.append(f" | {prefix}pooled_nr={metrics['pooled_norm_ratio']:.3f}")
    if "prompt_norm_ratio_distance" in metrics:
        parts.append(
            f" | {prefix}prompt_nrd={metrics['prompt_norm_ratio_distance']:.3f}"
        )
        parts.append(
            f" | {prefix}pooled_nrd={metrics['pooled_norm_ratio_distance']:.3f}"
        )
    if "prompt_std_ratio" in metrics:
        parts.append(f" | {prefix}prompt_sr={metrics['prompt_std_ratio']:.3f}")
        parts.append(f" | {prefix}pooled_sr={metrics['pooled_std_ratio']:.3f}")
    if "prompt_std_ratio_distance" in metrics:
        parts.append(
            f" | {prefix}prompt_srd={metrics['prompt_std_ratio_distance']:.3f}"
        )
        parts.append(
            f" | {prefix}pooled_srd={metrics['pooled_std_ratio_distance']:.3f}"
        )
    return "".join(parts)


def _instrumentation_extra_text(metrics, prefix=""):
    if "avg_batch_rows" not in metrics:
        return ""
    return (
        f" | {prefix}batch_rows={metrics['avg_batch_rows']:.1f}"
        f" | {prefix}chunk_rows={metrics['avg_chunk_rows']:.1f}"
        f" | {prefix}chunk_wait_ms={metrics['avg_chunk_wait_ms']:.1f}"
        f" | {prefix}copy_ms={metrics['avg_copy_ms']:.1f}"
        f" | {prefix}gpu_ms={metrics['avg_gpu_ms']:.1f}"
    )


def _format_epoch_metrics(prefix, metrics):
    return (
        f"{prefix}mse={metrics['mse']:.6f} | "
        f"{prefix}mae={metrics['mae']:.6f} | "
        f"{prefix}cos={metrics['cosine']:.6f}"
        f"{_metrics_extra_text(metrics, prefix=prefix)}"
        f"{_instrumentation_extra_text(metrics, prefix=prefix)}"
    )


def _compute_monitor_metric(target_family, metrics, sdxl_monitor_config=None):
    if target_family != "sdxl" or metrics is None or sdxl_monitor_config is None:
        return metrics["mse"], "mse"
    if sdxl_monitor_config.metric_name == "mse":
        return metrics["mse"], "mse"

    norm_distance = (
        metrics["prompt_norm_ratio_distance"] + metrics["pooled_norm_ratio_distance"]
    )
    std_distance = (
        metrics["prompt_std_ratio_distance"] + metrics["pooled_std_ratio_distance"]
    )
    composite = (
        metrics["mse"]
        + sdxl_monitor_config.norm_ratio_weight * norm_distance
        + sdxl_monitor_config.std_ratio_weight * std_distance
    )
    return composite, "composite"


def _compute_loss_metrics(
    predictions,
    target_bundle,
    target_family,
    criterion,
    prompt_loss_weight=1.0,
    pooled_loss_weight=1.0,
    sdxl_loss_config=None,
):
    if target_family == "sdxl":
        sdxl_loss_config = sdxl_loss_config or SDXLossConfig()
        prompt_predictions, pooled_predictions = predictions
        prompt_targets = target_bundle["prompt_embeds"]
        pooled_targets = target_bundle["pooled_prompt_embeds"]

        prompt_predictions_f32 = prompt_predictions.float()
        pooled_predictions_f32 = pooled_predictions.float()
        prompt_targets_f32 = prompt_targets.float()
        pooled_targets_f32 = pooled_targets.float()

        prompt_predictions_flat = _flatten_for_cosine(prompt_predictions_f32)
        prompt_targets_flat = _flatten_for_cosine(prompt_targets_f32)

        prompt_pointwise = criterion(prompt_predictions_f32, prompt_targets_f32)
        pooled_pointwise = criterion(pooled_predictions_f32, pooled_targets_f32)
        prompt_mse = F.mse_loss(prompt_predictions_f32, prompt_targets_f32)
        pooled_mse = F.mse_loss(pooled_predictions_f32, pooled_targets_f32)

        prompt_prediction_norms = torch.linalg.vector_norm(
            prompt_predictions_flat, dim=-1
        )
        prompt_target_norms = torch.linalg.vector_norm(prompt_targets_flat, dim=-1)
        pooled_prediction_norms = torch.linalg.vector_norm(
            pooled_predictions_f32, dim=-1
        )
        pooled_target_norms = torch.linalg.vector_norm(pooled_targets_f32, dim=-1)

        prompt_norm_loss = F.mse_loss(
            prompt_prediction_norms.clamp_min(1e-6).log(),
            prompt_target_norms.clamp_min(1e-6).log(),
        )
        pooled_norm_loss = F.mse_loss(
            pooled_prediction_norms.clamp_min(1e-6).log(),
            pooled_target_norms.clamp_min(1e-6).log(),
        )
        prompt_norm_match_loss = F.mse_loss(
            prompt_prediction_norms,
            prompt_target_norms,
        )
        pooled_norm_match_loss = F.mse_loss(
            pooled_prediction_norms,
            pooled_target_norms,
        )

        prompt_prediction_stds = prompt_predictions_flat.std(dim=-1, unbiased=False)
        prompt_target_stds = prompt_targets_flat.std(dim=-1, unbiased=False)
        pooled_prediction_stds = pooled_predictions_f32.std(dim=-1, unbiased=False)
        pooled_target_stds = pooled_targets_f32.std(dim=-1, unbiased=False)

        prompt_norm_ratio = (
            prompt_prediction_norms.mean() / prompt_target_norms.mean().clamp_min(1e-6)
        )
        pooled_norm_ratio = (
            pooled_prediction_norms.mean() / pooled_target_norms.mean().clamp_min(1e-6)
        )
        prompt_std_ratio = (
            prompt_prediction_stds.mean() / prompt_target_stds.mean().clamp_min(1e-6)
        )
        pooled_std_ratio = (
            pooled_prediction_stds.mean() / pooled_target_stds.mean().clamp_min(1e-6)
        )

        prompt_scale = (
            prompt_target_norms / prompt_prediction_norms.clamp_min(1e-6)
        ).unsqueeze(-1)
        pooled_scale = (
            pooled_target_norms / pooled_prediction_norms.clamp_min(1e-6)
        ).unsqueeze(-1)
        prompt_scale_matched_predictions = (
            prompt_predictions_flat * prompt_scale
        ).reshape_as(prompt_predictions_f32)
        pooled_scale_matched_predictions = pooled_predictions_f32 * pooled_scale
        prompt_scale_matched_mse = F.mse_loss(
            prompt_scale_matched_predictions, prompt_targets_f32
        )
        pooled_scale_matched_mse = F.mse_loss(
            pooled_scale_matched_predictions, pooled_targets_f32
        )

        prompt_cosine_value = F.cosine_similarity(
            prompt_predictions_flat,
            prompt_targets_flat,
            dim=-1,
        ).mean()
        pooled_cosine_value = F.cosine_similarity(
            pooled_predictions_f32,
            pooled_targets_f32,
            dim=-1,
        ).mean()

        prompt_objective = (
            prompt_pointwise
            + sdxl_loss_config.prompt_cosine_weight * (1.0 - prompt_cosine_value)
            + sdxl_loss_config.prompt_norm_weight * prompt_norm_loss
            + sdxl_loss_config.prompt_norm_match_weight * prompt_norm_match_loss
        )
        pooled_objective = (
            pooled_pointwise
            + sdxl_loss_config.pooled_cosine_weight * (1.0 - pooled_cosine_value)
            + sdxl_loss_config.pooled_norm_weight * pooled_norm_loss
            + sdxl_loss_config.pooled_norm_match_weight * pooled_norm_match_loss
        )

        loss = _weighted_average(
            [prompt_objective, pooled_objective],
            [prompt_loss_weight, pooled_loss_weight],
        )

        prompt_mae = (prompt_predictions_f32 - prompt_targets_f32).abs().mean().detach()
        pooled_mae = (pooled_predictions_f32 - pooled_targets_f32).abs().mean().detach()
        prompt_cosine = prompt_cosine_value.detach()
        pooled_cosine = pooled_cosine_value.detach()

        return loss, {
            "objective": loss.detach().float(),
            "pointwise": _weighted_average(
                [prompt_pointwise.detach().float(), pooled_pointwise.detach().float()],
                [prompt_loss_weight, pooled_loss_weight],
            ),
            "mse": _weighted_average(
                [prompt_mse.detach().float(), pooled_mse.detach().float()],
                [prompt_loss_weight, pooled_loss_weight],
            ),
            "mae": _weighted_average(
                [prompt_mae, pooled_mae],
                [prompt_loss_weight, pooled_loss_weight],
            ),
            "cosine": _weighted_average(
                [prompt_cosine, pooled_cosine],
                [prompt_loss_weight, pooled_loss_weight],
            ),
            "scale_matched_mse": _weighted_average(
                [
                    prompt_scale_matched_mse.detach().float(),
                    pooled_scale_matched_mse.detach().float(),
                ],
                [prompt_loss_weight, pooled_loss_weight],
            ),
            "prompt_mse": prompt_mse.detach().float(),
            "prompt_mae": prompt_mae,
            "prompt_cosine": prompt_cosine,
            "prompt_norm_loss": prompt_norm_loss.detach().float(),
            "prompt_norm_match_loss": prompt_norm_match_loss.detach().float(),
            "prompt_norm_ratio": prompt_norm_ratio.detach(),
            "prompt_norm_ratio_distance": (prompt_norm_ratio - 1.0).abs().detach(),
            "prompt_std_ratio": prompt_std_ratio.detach(),
            "prompt_std_ratio_distance": (prompt_std_ratio - 1.0).abs().detach(),
            "prompt_scale_matched_mse": prompt_scale_matched_mse.detach().float(),
            "pooled_mse": pooled_mse.detach().float(),
            "pooled_mae": pooled_mae,
            "pooled_cosine": pooled_cosine,
            "pooled_norm_loss": pooled_norm_loss.detach().float(),
            "pooled_norm_match_loss": pooled_norm_match_loss.detach().float(),
            "pooled_norm_ratio": pooled_norm_ratio.detach(),
            "pooled_norm_ratio_distance": (pooled_norm_ratio - 1.0).abs().detach(),
            "pooled_std_ratio": pooled_std_ratio.detach(),
            "pooled_std_ratio_distance": (pooled_std_ratio - 1.0).abs().detach(),
            "pooled_scale_matched_mse": pooled_scale_matched_mse.detach().float(),
        }

    targets = target_bundle["embedding"]
    loss = criterion(predictions, targets)
    predictions_f32 = predictions.float()
    targets_f32 = targets.float()
    return loss, {
        "objective": loss.detach().float(),
        "mse": loss.detach().float(),
        "mae": (predictions_f32 - targets_f32).abs().mean().detach(),
        "cosine": (
            F.cosine_similarity(predictions_f32, targets_f32, dim=-1).mean().detach()
        ),
    }


def _emit_progress(progress_callback, payload):
    if progress_callback is not None:
        progress_callback(payload)
        return

    total_steps = payload.get("total_steps") or 0
    total_rows = payload.get("total_rows") or 0
    step_text = (
        f"{payload['step']}/{total_steps}" if total_steps else str(payload["step"])
    )
    rows_text = (
        f"{payload['rows']:,}/{total_rows:,}" if total_rows else f"{payload['rows']:,}"
    )
    lr = payload.get("lr")
    lr_text = "" if lr is None else f" | lr={lr:.6e}"
    extra_text = _metrics_extra_text(payload)
    print(
        f"{payload['split'].upper()} progress | "
        f"epoch={payload['epoch']}/{payload['epochs']} | "
        f"step={step_text} | rows={rows_text} | "
        f"mse={payload['mse']:.6f} | mae={payload['mae']:.6f} | "
        f"cos={payload['cosine']:.6f} | rate={payload['rows_per_s']:.1f} rows/s"
        f"{lr_text}{extra_text}{_instrumentation_extra_text(payload)}",
        flush=True,
    )


def _split_mask_from_indices(indices, val_fraction, seed):
    if val_fraction <= 0.0:
        return np.zeros(len(indices), dtype=bool)

    threshold = int(round(val_fraction * 10_000))
    if threshold <= 0:
        return np.zeros(len(indices), dtype=bool)
    if threshold >= 10_000:
        return np.ones(len(indices), dtype=bool)

    hashed = (
        indices.astype(np.uint64) * np.uint64(11400714819323198485) + np.uint64(seed)
    ) % np.uint64(10_000)
    return hashed < np.uint64(threshold)


OUTPUT_DIR = _env_str("TARGET_OUTPUT_DIR", "embedded_chunks")
QWEN_OUTPUT_DIR = _env_str(
    "QWEN_OUTPUT_DIR", _env_str("SOURCE_OUTPUT_DIR", "embedded_chunks")
)
QWEN_ARCHIVE_DIR = os.path.join(QWEN_OUTPUT_DIR, "archive")
CLIP_ARCHIVE_DIR = os.path.join(OUTPUT_DIR, "clip_archive")


def _list_npz_archives(archive_dir, prefix):
    if not os.path.exists(archive_dir):
        return []
    return sorted(
        os.path.join(archive_dir, f)
        for f in os.listdir(archive_dir)
        if f.startswith(prefix) and f.endswith(".npz")
    )


def _resolve_target_archives(target_layout=None, clip_archive_dir=CLIP_ARCHIVE_DIR):
    if target_layout is None:
        return _list_npz_archives(clip_archive_dir, "clip_archive_"), clip_archive_dir

    archives = _list_npz_archives(
        target_layout.archive_dir, target_layout.archive_prefix
    )
    if archives:
        return archives, target_layout.archive_dir

    if (
        target_layout.legacy_archive_dir is not None
        and target_layout.legacy_archive_prefix is not None
    ):
        legacy_archives = _list_npz_archives(
            target_layout.legacy_archive_dir,
            target_layout.legacy_archive_prefix,
        )
        if legacy_archives:
            return legacy_archives, target_layout.legacy_archive_dir

    return [], target_layout.archive_dir


def _load_target_bundle_from_archive(data, target_family):
    if target_family == "sdxl":
        if "prompt_embeds" not in data or "pooled_prompt_embeds" not in data:
            raise RuntimeError(
                "Expected SDXL target archive with 'prompt_embeds' and 'pooled_prompt_embeds' arrays"
            )
        return {
            "prompt_embeds": np.asarray(data["prompt_embeds"]),
            "pooled_prompt_embeds": np.asarray(data["pooled_prompt_embeds"]),
        }

    if "embeddings" not in data:
        raise RuntimeError("Expected target archive with 'embeddings' array")
    return {"embedding": np.asarray(data["embeddings"])}


def _allocate_target_buffers(total_rows, sample_bundle):
    return {
        key: np.empty((total_rows, *array.shape[1:]), dtype=array.dtype)
        for key, array in sample_bundle.items()
    }


def _fill_target_buffers(archives, total_rows, target_family, indices_buffer, buffers):
    offset = 0
    for path in archives:
        if offset >= total_rows:
            break
        with np.load(path, allow_pickle=False) as data:
            bundle = _load_target_bundle_from_archive(data, target_family)
            take = min(total_rows - offset, len(data["indices"]))
            indices_buffer[offset : offset + take] = data["indices"][:take]
            for key, array in bundle.items():
                buffers[key][offset : offset + take] = array[:take]
            offset += take
    return offset


def _qwen_archives_have_sequence_inputs(qwen_archive_dir):
    archives = _list_npz_archives(qwen_archive_dir, "archive_")
    if not archives:
        return False
    with np.load(archives[0], allow_pickle=False) as data:
        return "token_embeddings" in data and "token_offsets" in data


def _load_qwen_bundle_from_archive(data):
    if "token_embeddings" in data and "token_offsets" in data:
        token_embeddings = np.asarray(data["token_embeddings"], dtype=np.float32)
        token_offsets = np.asarray(data["token_offsets"], dtype=np.int64)
        if "pooled_embeddings" in data:
            pooled_embeddings = np.asarray(data["pooled_embeddings"], dtype=np.float32)
        else:
            pooled_rows = []
            for row_idx in range(len(token_offsets) - 1):
                start = int(token_offsets[row_idx])
                end = int(token_offsets[row_idx + 1])
                pooled_rows.append(token_embeddings[start:end].mean(axis=0))
            pooled_embeddings = np.stack(pooled_rows).astype(np.float32, copy=False)
        if "sequence_lengths" in data:
            sequence_lengths = np.asarray(data["sequence_lengths"], dtype=np.int32)
        else:
            sequence_lengths = np.diff(token_offsets).astype(np.int32, copy=False)
        return {
            "format": "token_sequence",
            "pooled_embeddings": pooled_embeddings,
            "token_embeddings": token_embeddings,
            "token_offsets": token_offsets,
            "sequence_lengths": sequence_lengths,
        }

    if "embeddings" not in data:
        raise RuntimeError(
            "Expected Qwen archive with either 'embeddings' or token-sequence arrays."
        )
    return np.asarray(data["embeddings"], dtype=np.float32)


def _qwen_rows(bundle):
    if isinstance(bundle, dict):
        return int(bundle["pooled_embeddings"].shape[0])
    return int(bundle.shape[0])


def _qwen_input_dim(bundle):
    if isinstance(bundle, dict):
        return int(bundle["pooled_embeddings"].shape[1])
    return int(bundle.shape[1])


def _slice_qwen_bundle_np(bundle, start, end):
    if isinstance(bundle, np.ndarray):
        return bundle[start:end]

    token_start = int(bundle["token_offsets"][start])
    token_end = int(bundle["token_offsets"][end])
    return {
        "format": "token_sequence",
        "pooled_embeddings": bundle["pooled_embeddings"][start:end],
        "token_embeddings": bundle["token_embeddings"][token_start:token_end],
        "token_offsets": bundle["token_offsets"][start : end + 1] - token_start,
        "sequence_lengths": bundle["sequence_lengths"][start:end],
    }


def _qwen_bundle_from_numpy(bundle):
    if isinstance(bundle, np.ndarray):
        return torch.from_numpy(bundle)
    return {
        "format": bundle["format"],
        "pooled_embeddings": torch.from_numpy(bundle["pooled_embeddings"]),
        "token_embeddings": torch.from_numpy(bundle["token_embeddings"]),
        "token_offsets": torch.from_numpy(bundle["token_offsets"]),
        "sequence_lengths": torch.from_numpy(
            bundle["sequence_lengths"].astype(np.int64, copy=False)
        ),
    }


def _select_qwen_bundle(bundle, indices):
    if isinstance(bundle, torch.Tensor):
        return bundle.index_select(0, indices)

    pooled_embeddings = bundle["pooled_embeddings"].index_select(0, indices)
    sequence_lengths = bundle["sequence_lengths"].index_select(0, indices)
    selected_indices = indices.tolist()
    token_segments = []
    new_offsets = [0]
    for row_idx in selected_indices:
        token_start = int(bundle["token_offsets"][row_idx].item())
        token_end = int(bundle["token_offsets"][row_idx + 1].item())
        token_segments.append(bundle["token_embeddings"][token_start:token_end])
        new_offsets.append(new_offsets[-1] + (token_end - token_start))
    if token_segments:
        token_embeddings = torch.cat(token_segments, dim=0)
    else:
        token_embeddings = bundle["token_embeddings"].new_empty(
            (0, bundle["token_embeddings"].shape[-1])
        )
    return {
        "format": "token_sequence",
        "pooled_embeddings": pooled_embeddings,
        "token_embeddings": token_embeddings,
        "token_offsets": torch.tensor(new_offsets, dtype=torch.int64),
        "sequence_lengths": sequence_lengths,
    }


def _concat_qwen_bundles(*bundles):
    active = [bundle for bundle in bundles if bundle is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    first = active[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(active, dim=0)

    pooled_embeddings = torch.cat(
        [bundle["pooled_embeddings"] for bundle in active], dim=0
    )
    token_embeddings = torch.cat(
        [bundle["token_embeddings"] for bundle in active], dim=0
    )
    sequence_lengths = torch.cat(
        [bundle["sequence_lengths"] for bundle in active], dim=0
    )
    token_offsets = [torch.zeros(1, dtype=torch.int64)]
    total_tokens = 0
    for bundle in active:
        offsets = bundle["token_offsets"][1:] + total_tokens
        token_offsets.append(offsets)
        if len(offsets) > 0:
            total_tokens = int(offsets[-1].item())
    return {
        "format": "token_sequence",
        "pooled_embeddings": pooled_embeddings,
        "token_embeddings": token_embeddings,
        "token_offsets": torch.cat(token_offsets, dim=0),
        "sequence_lengths": sequence_lengths,
    }


def _slice_qwen_bundle(bundle, start, end):
    if isinstance(bundle, torch.Tensor):
        return bundle[start:end]

    token_start = int(bundle["token_offsets"][start].item())
    token_end = int(bundle["token_offsets"][end].item())
    return {
        "format": "token_sequence",
        "pooled_embeddings": bundle["pooled_embeddings"][start:end].clone(),
        "token_embeddings": bundle["token_embeddings"][token_start:token_end].clone(),
        "token_offsets": bundle["token_offsets"][start : end + 1].clone() - token_start,
        "sequence_lengths": bundle["sequence_lengths"][start:end].clone(),
    }


def _pin_qwen_bundle(bundle):
    if isinstance(bundle, torch.Tensor):
        return bundle.pin_memory()
    return {
        key: value.pin_memory() if torch.is_tensor(value) else value
        for key, value in bundle.items()
    }


def _move_qwen_bundle_to_device(bundle, device, pin_memory):
    if isinstance(bundle, torch.Tensor):
        return bundle.to(device, non_blocking=pin_memory)
    return {
        key: (
            value.to(device, non_blocking=pin_memory)
            if torch.is_tensor(value)
            else value
        )
        for key, value in bundle.items()
    }


def _materialize_qwen_batch(bundle):
    if isinstance(bundle, torch.Tensor):
        return bundle

    batch_size = int(bundle["pooled_embeddings"].shape[0])
    if batch_size == 0:
        hidden_dim = int(bundle["token_embeddings"].shape[-1])
        return {
            "pooled_embeddings": bundle["pooled_embeddings"],
            "token_embeddings": bundle["token_embeddings"].new_empty(
                (0, 0, hidden_dim)
            ),
            "attention_mask": torch.zeros((0, 0), dtype=torch.bool),
            "sequence_lengths": bundle["sequence_lengths"],
        }

    max_tokens = int(bundle["sequence_lengths"].max().item())
    hidden_dim = int(bundle["pooled_embeddings"].shape[1])
    token_matrix = bundle["pooled_embeddings"].new_zeros(
        (batch_size, max_tokens, hidden_dim)
    )
    attention_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    for row_idx in range(batch_size):
        token_start = int(bundle["token_offsets"][row_idx].item())
        token_end = int(bundle["token_offsets"][row_idx + 1].item())
        token_count = token_end - token_start
        if token_count <= 0:
            continue
        token_matrix[row_idx, :token_count] = bundle["token_embeddings"][
            token_start:token_end
        ]
        attention_mask[row_idx, :token_count] = True
    return {
        "pooled_embeddings": bundle["pooled_embeddings"],
        "token_embeddings": token_matrix,
        "attention_mask": attention_mask,
        "sequence_lengths": bundle["sequence_lengths"],
    }


def _forward_qwen_model(model, batch_qwen):
    if isinstance(batch_qwen, torch.Tensor):
        return model(batch_qwen)
    if getattr(model, "expects_qwen_sequences", False):
        return model(batch_qwen)
    return model(batch_qwen["pooled_embeddings"])


# ==========================================
# 1. DEFINE THE NEURAL NETWORK ARCHITECTURE
# ==========================================
class QwenToSDProjector(nn.Module):
    # Dimensions are inferred from the loaded training data.
    def __init__(self, qwen_dim, sd_dim, hidden_dim=4096):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(qwen_dim, hidden_dim),  # Layer 1 (proj_w1, proj_b1)
            nn.GELU(),
            nn.Linear(hidden_dim, sd_dim),  # Layer 2 (proj_w2, proj_b2)
        )

    def forward(self, x):
        return self.projection(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim, residual=True):
        super().__init__()
        self.residual = residual
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        update = self.fc2(self.act(self.fc1(x)))
        if self.residual:
            return x + update
        return update


class OutputAffineCalibrator(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x):
        return x * self.gain + self.bias


class QwenToSDXLProjector(nn.Module):
    architecture_name = "mlp"

    def __init__(
        self,
        qwen_dim,
        prompt_seq_len=77,
        prompt_dim=2048,
        pooled_dim=1280,
        hidden_dim=4096,
        prompt_token_dim=256,
        trunk_depth=1,
        residual_trunk=True,
        prompt_head_hidden_dim=512,
        pooled_head_hidden_dim=2048,
        use_output_calibrator=False,
    ):
        super().__init__()
        self.qwen_dim = qwen_dim
        self.hidden_dim = hidden_dim
        self.prompt_seq_len = prompt_seq_len
        self.prompt_dim = prompt_dim
        self.pooled_dim = pooled_dim
        self.prompt_token_dim = prompt_token_dim
        self.trunk_depth = max(0, int(trunk_depth))
        self.residual_trunk = bool(residual_trunk)
        self.prompt_head_hidden_dim = prompt_head_hidden_dim
        self.pooled_head_hidden_dim = pooled_head_hidden_dim
        self.use_output_calibrator = bool(use_output_calibrator)

        self.input_projection = nn.Sequential(
            nn.Linear(qwen_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.trunk_layers = nn.ModuleList(
            ResidualMLPBlock(hidden_dim, residual=self.residual_trunk)
            for _ in range(self.trunk_depth)
        )
        self.prompt_seed = nn.Linear(hidden_dim, prompt_seq_len * prompt_token_dim)
        self.prompt_projection = nn.Sequential(
            nn.GELU(),
            nn.Linear(prompt_token_dim, prompt_head_hidden_dim),
            nn.GELU(),
            nn.Linear(prompt_head_hidden_dim, prompt_dim),
        )
        self.pooled_head = nn.Sequential(
            nn.Linear(hidden_dim, pooled_head_hidden_dim),
            nn.GELU(),
            nn.Linear(pooled_head_hidden_dim, pooled_dim),
        )
        if self.use_output_calibrator:
            self.prompt_output_calibrator = OutputAffineCalibrator(prompt_dim)
            self.pooled_output_calibrator = OutputAffineCalibrator(pooled_dim)
        else:
            self.prompt_output_calibrator = None
            self.pooled_output_calibrator = None

    def forward(self, x):
        trunk_features = self.input_projection(x)
        for block in self.trunk_layers:
            trunk_features = block(trunk_features)
        prompt_seed = self.prompt_seed(trunk_features).view(
            x.shape[0], self.prompt_seq_len, self.prompt_token_dim
        )
        prompt_out = self.prompt_projection(prompt_seed)
        pooled_out = self.pooled_head(trunk_features)
        if self.prompt_output_calibrator is not None:
            prompt_out = self.prompt_output_calibrator(prompt_out)
            pooled_out = self.pooled_output_calibrator(pooled_out)
        return prompt_out, pooled_out


class QwenResamplerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.latents_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, latents, context, key_padding_mask=None):
        attn_latents = self.latents_norm(latents)
        attn_context = self.context_norm(context)
        attn_output, _ = self.cross_attn(
            attn_latents,
            attn_context,
            attn_context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latents = latents + attn_output
        latents = latents + self.ff(self.ff_norm(latents))
        return latents


class QwenTokenToSDXLResampler(nn.Module):
    expects_qwen_sequences = True
    architecture_name = "resampler"

    def __init__(
        self,
        qwen_dim,
        prompt_seq_len=77,
        prompt_dim=2048,
        pooled_dim=1280,
        hidden_dim=4096,
        resampler_depth=2,
        resampler_heads=8,
        resampler_ff_dim=16384,
        pooled_query_count=1,
        prompt_head_hidden_dim=512,
        pooled_head_hidden_dim=2048,
    ):
        super().__init__()
        if hidden_dim % resampler_heads != 0:
            raise ValueError(
                "TRAIN_HIDDEN_DIM must be divisible by TRAIN_SDXL_RESAMPLER_HEADS "
                f"(got hidden_dim={hidden_dim}, heads={resampler_heads})."
            )
        if pooled_query_count < 1:
            raise ValueError("TRAIN_SDXL_RESAMPLER_POOLED_QUERIES must be >= 1.")

        self.qwen_dim = qwen_dim
        self.hidden_dim = hidden_dim
        self.prompt_seq_len = prompt_seq_len
        self.prompt_dim = prompt_dim
        self.pooled_dim = pooled_dim
        self.resampler_depth = max(1, int(resampler_depth))
        self.resampler_heads = int(resampler_heads)
        self.resampler_ff_dim = int(resampler_ff_dim)
        self.pooled_query_count = int(pooled_query_count)
        self.prompt_head_hidden_dim = int(prompt_head_hidden_dim)
        self.pooled_head_hidden_dim = int(pooled_head_hidden_dim)

        self.input_projection = nn.Sequential(
            nn.Linear(qwen_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.latent_queries = nn.Parameter(
            torch.empty(1, prompt_seq_len + pooled_query_count, hidden_dim)
        )
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)

        self.resampler_blocks = nn.ModuleList(
            QwenResamplerBlock(
                hidden_dim=hidden_dim,
                num_heads=self.resampler_heads,
                ff_dim=self.resampler_ff_dim,
            )
            for _ in range(self.resampler_depth)
        )
        self.prompt_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.prompt_head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.prompt_head_hidden_dim, prompt_dim),
        )
        self.pooled_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.pooled_head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.pooled_head_hidden_dim, pooled_dim),
        )

    def forward(self, qwen_inputs):
        token_embeddings = qwen_inputs["token_embeddings"]
        attention_mask = qwen_inputs.get("attention_mask")
        if token_embeddings.ndim != 3:
            raise RuntimeError(
                "Resampler expects token_embeddings with shape [batch, tokens, hidden]."
            )
        if (
            attention_mask is not None
            and attention_mask.shape[:2] != token_embeddings.shape[:2]
        ):
            raise RuntimeError(
                "attention_mask must match token_embeddings batch and token dimensions."
            )

        context = self.input_projection(token_embeddings)
        latents = self.latent_queries.expand(context.shape[0], -1, -1)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        for block in self.resampler_blocks:
            latents = block(latents, context, key_padding_mask=key_padding_mask)

        prompt_latents = latents[:, : self.prompt_seq_len]
        pooled_latents = latents[:, self.prompt_seq_len :]
        pooled_source = (
            pooled_latents[:, 0]
            if self.pooled_query_count == 1
            else pooled_latents.mean(dim=1)
        )
        return self.prompt_head(prompt_latents), self.pooled_head(pooled_source)


def _resolve_sdxl_projector_architecture(requested_architecture, has_sequence_inputs):
    architecture = (requested_architecture or "mlp").strip().lower()
    if architecture not in {"auto", "mlp", "resampler"}:
        raise ValueError(
            "TRAIN_SDXL_PROJECTOR_ARCH must be one of: auto, mlp, resampler. "
            f"Got '{requested_architecture}'."
        )
    if architecture == "auto":
        return "resampler" if has_sequence_inputs else "mlp"
    return architecture


def _build_sdxl_projector(
    architecture,
    *,
    qwen_dim,
    prompt_seq_len,
    prompt_dim,
    pooled_dim,
    hidden_dim,
    prompt_token_dim,
    trunk_depth,
    residual_trunk,
    prompt_head_hidden_dim,
    pooled_head_hidden_dim,
    resampler_depth,
    resampler_heads,
    resampler_ff_mult,
    pooled_query_count,
    has_sequence_inputs,
    use_output_calibrator,
):
    if architecture == "resampler":
        if not has_sequence_inputs:
            raise ValueError(
                "The resampler architecture requires token-sequence Qwen archives. "
                "Re-extract Qwen embeddings without mean pooling before selecting TRAIN_SDXL_PROJECTOR_ARCH=resampler."
            )
        resampler_ff_mult = max(1, int(resampler_ff_mult))
        return QwenTokenToSDXLResampler(
            qwen_dim=qwen_dim,
            prompt_seq_len=prompt_seq_len,
            prompt_dim=prompt_dim,
            pooled_dim=pooled_dim,
            hidden_dim=hidden_dim,
            resampler_depth=resampler_depth,
            resampler_heads=resampler_heads,
            resampler_ff_dim=hidden_dim * resampler_ff_mult,
            pooled_query_count=pooled_query_count,
            prompt_head_hidden_dim=prompt_head_hidden_dim,
            pooled_head_hidden_dim=pooled_head_hidden_dim,
        )

    return QwenToSDXLProjector(
        qwen_dim=qwen_dim,
        prompt_seq_len=prompt_seq_len,
        prompt_dim=prompt_dim,
        pooled_dim=pooled_dim,
        hidden_dim=hidden_dim,
        prompt_token_dim=prompt_token_dim,
        trunk_depth=trunk_depth,
        residual_trunk=residual_trunk,
        prompt_head_hidden_dim=prompt_head_hidden_dim,
        pooled_head_hidden_dim=pooled_head_hidden_dim,
        use_output_calibrator=use_output_calibrator,
    )


def _sdxl_architecture_summary(model):
    if getattr(model, "architecture_name", "mlp") == "resampler":
        return (
            "SDXL architecture: "
            f"arch=resampler | heads={model.resampler_heads} | "
            f"depth={model.resampler_depth} | ff_dim={model.resampler_ff_dim} | "
            f"pooled_queries={model.pooled_query_count} | "
            f"prompt_head_dim={model.prompt_head_hidden_dim} | "
            f"pooled_head_dim={model.pooled_head_hidden_dim}"
        )
    return (
        "SDXL architecture: "
        f"arch=mlp | prompt_token_dim={model.prompt_token_dim} | "
        f"trunk_depth={model.trunk_depth} | "
        f"residual_trunk={model.residual_trunk} | "
        f"output_calibrator={model.use_output_calibrator} | "
        f"prompt_head_dim={model.prompt_head_hidden_dim} | "
        f"pooled_head_dim={model.pooled_head_hidden_dim}"
    )


# ==========================================
# 2. REAL DATA LOADING (PARQUET INGESTION)
# ==========================================
class ParquetEmbeddingDataset(Dataset):
    def __init__(self, parquet_path, sd_dim=1024, max_samples=None):
        print(f"Loading data from {parquet_path}...")
        df = pd.read_parquet(parquet_path)

        if "clip_embedding" in df.columns:
            df = df.dropna(subset=["qwen_embedding", "clip_embedding"])
        else:
            df = df.dropna(subset=["qwen_embedding"])

        if max_samples is not None:
            df = df.iloc[:max_samples]

        print("Packing Qwen embeddings...")
        qwen_embeds = np.stack(df["qwen_embedding"].to_numpy()).astype(
            np.float32, copy=False
        )
        self.inputs = torch.from_numpy(qwen_embeds)
        self.input_dim = int(self.inputs.shape[1])

        if "clip_embedding" in df.columns:
            print("Packing target CLIP embeddings...")
            clip_embeds = np.stack(df["clip_embedding"].to_numpy()).astype(
                np.float32, copy=False
            )
            self.targets = torch.from_numpy(clip_embeds)
            self.target_dim = int(self.targets.shape[1])
            print(
                f"Loaded {len(self.inputs):,} aligned training pairs "
                f"({self.input_dim} -> {self.target_dim})."
            )
        else:
            print("\n⚠️  WARNING: 'clip_embedding' column not found!")
            print("Generating mock target tensors so the sanity check can run.\n")
            self.targets = torch.randn(len(self.inputs), sd_dim, dtype=torch.float32)
            self.target_dim = sd_dim

        self.target_family = "sd"

        del df

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], {"embedding": self.targets[idx]}


class ArchiveEmbeddingDataset(Dataset):
    def __init__(
        self,
        qwen_archive_dir=QWEN_ARCHIVE_DIR,
        clip_archive_dir=CLIP_ARCHIVE_DIR,
        target_layout=None,
        max_samples=None,
    ):
        self.target_family = target_layout.family if target_layout is not None else "sd"
        qwen_archives = _list_npz_archives(qwen_archive_dir, "archive_")
        clip_archives, resolved_target_dir = _resolve_target_archives(
            target_layout, clip_archive_dir
        )

        if not qwen_archives:
            raise FileNotFoundError(f"No Qwen archives found in {qwen_archive_dir}")
        if not clip_archives:
            raise FileNotFoundError(
                f"No target archives found in {resolved_target_dir}"
            )

        print(f"Loading Qwen archives from {qwen_archive_dir}...")
        qwen_total = 0
        qwen_dim = None
        qwen_has_sequence_inputs = False
        for path in qwen_archives:
            with np.load(path, allow_pickle=False) as data:
                qwen_total += len(data["indices"])
                if qwen_dim is None:
                    qwen_bundle = _load_qwen_bundle_from_archive(data)
                    qwen_dim = _qwen_input_dim(qwen_bundle)
                    qwen_has_sequence_inputs = isinstance(qwen_bundle, dict)

        if qwen_has_sequence_inputs:
            raise NotImplementedError(
                "Token-sequence Qwen archives currently require streaming archive training. "
                "Set TRAIN_ARCHIVE_IN_MEMORY_LIMIT low enough to force the streaming path."
            )

        print(f"Loading target archives from {resolved_target_dir}...")
        clip_total = 0
        sample_bundle = None
        for path in clip_archives:
            with np.load(path, allow_pickle=False) as data:
                clip_total += len(data["indices"])
                if sample_bundle is None:
                    sample_bundle = _load_target_bundle_from_archive(
                        data, self.target_family
                    )

        if qwen_total != clip_total:
            raise RuntimeError(
                f"Archive row mismatch: {qwen_total:,} Qwen rows vs {clip_total:,} CLIP rows"
            )

        if sample_bundle is None:
            raise RuntimeError(
                f"No readable target archive payloads found in {resolved_target_dir}"
            )

        total_rows = qwen_total
        if max_samples is not None:
            total_rows = min(total_rows, max_samples)

        qwen_indices = np.empty(total_rows, dtype=np.int64)
        clip_indices = np.empty(total_rows, dtype=np.int64)
        qwen_embeds = np.empty((total_rows, qwen_dim), dtype=np.float32)
        target_buffers = _allocate_target_buffers(total_rows, sample_bundle)

        def _fill_buffers(archives, indices_buffer, embeddings_buffer):
            offset = 0
            for path in archives:
                if offset >= total_rows:
                    break
                data = np.load(path, allow_pickle=False)
                take = min(total_rows - offset, len(data["indices"]))
                indices_buffer[offset : offset + take] = data["indices"][:take]
                embeddings_buffer[offset : offset + take] = data["embeddings"][:take]
                offset += take
            return offset

        qwen_filled = _fill_buffers(qwen_archives, qwen_indices, qwen_embeds)
        clip_filled = _fill_target_buffers(
            clip_archives,
            total_rows,
            self.target_family,
            clip_indices,
            target_buffers,
        )
        if qwen_filled != total_rows or clip_filled != total_rows:
            raise RuntimeError(
                f"Archive load incomplete: qwen={qwen_filled:,}, clip={clip_filled:,}, expected={total_rows:,}"
            )
        if not np.array_equal(qwen_indices, clip_indices):
            raise RuntimeError("Qwen and CLIP archive indices do not align")

        self.inputs = torch.from_numpy(qwen_embeds)
        self.input_dim = qwen_dim
        if self.target_family == "sdxl":
            self.prompt_targets = torch.from_numpy(target_buffers["prompt_embeds"])
            self.pooled_targets = torch.from_numpy(
                target_buffers["pooled_prompt_embeds"]
            )
            self.prompt_seq_len = int(self.prompt_targets.shape[1])
            self.prompt_dim = int(self.prompt_targets.shape[2])
            self.pooled_dim = int(self.pooled_targets.shape[1])
            self.target_dim = {
                "prompt_seq_len": self.prompt_seq_len,
                "prompt_dim": self.prompt_dim,
                "pooled_dim": self.pooled_dim,
            }
            print(
                f"Loaded {len(self.inputs):,} aligned archive training pairs "
                f"({self.input_dim} -> [{self.prompt_seq_len}, {self.prompt_dim}] + {self.pooled_dim})."
            )
        else:
            self.targets = torch.from_numpy(target_buffers["embedding"])
            self.target_dim = int(self.targets.shape[1])
            print(
                f"Loaded {len(self.inputs):,} aligned archive training pairs "
                f"({self.input_dim} -> {self.target_dim})."
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.target_family == "sdxl":
            return self.inputs[idx], {
                "prompt_embeds": self.prompt_targets[idx],
                "pooled_prompt_embeds": self.pooled_targets[idx],
            }
        return self.inputs[idx], {"embedding": self.targets[idx]}


class ArchiveChunkReader:
    def __init__(
        self,
        qwen_archive_dir=QWEN_ARCHIVE_DIR,
        clip_archive_dir=CLIP_ARCHIVE_DIR,
        target_layout=None,
        max_samples=None,
        archive_threads=0,
    ):
        self.target_family = target_layout.family if target_layout is not None else "sd"
        self.archive_threads = max(0, int(archive_threads))
        self.qwen_archives = _list_npz_archives(qwen_archive_dir, "archive_")
        self.clip_archives, resolved_target_dir = _resolve_target_archives(
            target_layout, clip_archive_dir
        )
        if not self.qwen_archives:
            raise FileNotFoundError(f"No Qwen archives found in {qwen_archive_dir}")
        if not self.clip_archives:
            raise FileNotFoundError(
                f"No target archives found in {resolved_target_dir}"
            )

        self.qwen_index_arrays = []
        self.qwen_counts = []
        self.clip_counts = []
        self.input_dim = None
        self.target_dim = None
        self.prompt_seq_len = None
        self.prompt_dim = None
        self.pooled_dim = None
        self.has_sequence_inputs = False

        qwen_total = 0
        for path in self.qwen_archives:
            with np.load(path, allow_pickle=False) as data:
                indices = data["indices"].astype(np.int64, copy=False)
                self.qwen_index_arrays.append(indices)
                self.qwen_counts.append(len(indices))
                qwen_total += len(indices)
                if self.input_dim is None:
                    qwen_bundle = _load_qwen_bundle_from_archive(data)
                    self.input_dim = _qwen_input_dim(qwen_bundle)
                    self.has_sequence_inputs = isinstance(qwen_bundle, dict)

        clip_total = 0
        for path in self.clip_archives:
            with np.load(path, allow_pickle=False) as data:
                self.clip_counts.append(len(data["indices"]))
                clip_total += len(data["indices"])
                if self.target_dim is None:
                    bundle = _load_target_bundle_from_archive(data, self.target_family)
                    if self.target_family == "sdxl":
                        self.prompt_seq_len = int(bundle["prompt_embeds"].shape[1])
                        self.prompt_dim = int(bundle["prompt_embeds"].shape[2])
                        self.pooled_dim = int(bundle["pooled_prompt_embeds"].shape[1])
                        self.target_dim = {
                            "prompt_seq_len": self.prompt_seq_len,
                            "prompt_dim": self.prompt_dim,
                            "pooled_dim": self.pooled_dim,
                        }
                    else:
                        self.target_dim = int(bundle["embedding"].shape[1])

        if qwen_total != clip_total:
            raise RuntimeError(
                f"Archive row mismatch: {qwen_total:,} Qwen rows vs {clip_total:,} CLIP rows"
            )

        self.total_rows = qwen_total
        if max_samples is not None:
            self.total_rows = min(self.total_rows, max_samples)

    def summarize(self, val_fraction, seed, batch_size):
        train_rows = 0
        val_rows = 0

        q_file_idx = 0
        c_file_idx = 0
        q_pos = 0
        c_pos = 0
        rows_left = self.total_rows

        while rows_left > 0:
            q_indices = self.qwen_index_arrays[q_file_idx]
            q_rem = len(q_indices) - q_pos
            c_rem = self.clip_counts[c_file_idx] - c_pos
            take = min(rows_left, q_rem, c_rem)

            chunk_indices = q_indices[q_pos : q_pos + take]
            val_mask = _split_mask_from_indices(chunk_indices, val_fraction, seed)
            chunk_val_rows = int(val_mask.sum())
            chunk_train_rows = take - chunk_val_rows
            train_rows += chunk_train_rows
            val_rows += chunk_val_rows

            q_pos += take
            c_pos += take
            rows_left -= take
            if q_pos == len(q_indices):
                q_file_idx += 1
                q_pos = 0
            if c_pos == self.clip_counts[c_file_idx]:
                c_file_idx += 1
                c_pos = 0

        train_steps = math.ceil(train_rows / batch_size) if train_rows > 0 else 0
        val_steps = math.ceil(val_rows / batch_size) if val_rows > 0 else 0

        return {
            "train_rows": train_rows,
            "val_rows": val_rows,
            "train_steps": train_steps,
            "val_steps": val_steps,
        }

    def _load_qwen_archive(self, path):
        with np.load(path, allow_pickle=False) as data:
            return {
                "indices": data["indices"].astype(np.int64, copy=False),
                "inputs": _load_qwen_bundle_from_archive(data),
            }

    def _load_target_archive(self, path):
        with np.load(path, allow_pickle=False) as data:
            return {
                "indices": data["indices"].astype(np.int64, copy=False),
                "targets": _load_target_bundle_from_archive(data, self.target_family),
            }

    def iter_chunks(self):
        q_file_idx = 0
        c_file_idx = 0
        q_pos = 0
        c_pos = 0
        rows_left = self.total_rows
        q_current = self._load_qwen_archive(self.qwen_archives[q_file_idx])
        c_current = self._load_target_archive(self.clip_archives[c_file_idx])

        def _submit_prefetch(executor, paths, next_idx, loader):
            if executor is None or next_idx >= len(paths):
                return None
            return executor.submit(loader, paths[next_idx])

        executor = None
        if self.archive_threads > 0:
            executor = ThreadPoolExecutor(max_workers=min(self.archive_threads, 2))

        try:
            q_future = _submit_prefetch(
                executor, self.qwen_archives, q_file_idx + 1, self._load_qwen_archive
            )
            c_future = _submit_prefetch(
                executor, self.clip_archives, c_file_idx + 1, self._load_target_archive
            )

            while rows_left > 0:
                q_rem = len(q_current["indices"]) - q_pos
                c_rem = len(c_current["indices"]) - c_pos
                take = min(rows_left, q_rem, c_rem)
                q_indices = q_current["indices"][q_pos : q_pos + take]
                c_indices = c_current["indices"][c_pos : c_pos + take]
                if not np.array_equal(q_indices, c_indices):
                    raise RuntimeError("Qwen and CLIP archive indices do not align")

                yield (
                    q_indices,
                    _slice_qwen_bundle_np(q_current["inputs"], q_pos, q_pos + take),
                    {
                        key: value[c_pos : c_pos + take]
                        for key, value in c_current["targets"].items()
                    },
                )

                q_pos += take
                c_pos += take
                rows_left -= take

                if q_pos == len(q_current["indices"]):
                    q_file_idx += 1
                    q_pos = 0
                    if q_file_idx < len(self.qwen_archives):
                        if q_future is not None:
                            q_current = q_future.result()
                        else:
                            q_current = self._load_qwen_archive(
                                self.qwen_archives[q_file_idx]
                            )
                        q_future = _submit_prefetch(
                            executor,
                            self.qwen_archives,
                            q_file_idx + 1,
                            self._load_qwen_archive,
                        )
                if c_pos == len(c_current["indices"]):
                    c_file_idx += 1
                    c_pos = 0
                    if c_file_idx < len(self.clip_archives):
                        if c_future is not None:
                            c_current = c_future.result()
                        else:
                            c_current = self._load_target_archive(
                                self.clip_archives[c_file_idx]
                            )
                        c_future = _submit_prefetch(
                            executor,
                            self.clip_archives,
                            c_file_idx + 1,
                            self._load_target_archive,
                        )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)


def _split_dataset(dataset, val_fraction, seed):
    total_rows = len(dataset)
    if total_rows < 2 or val_fraction <= 0.0:
        return dataset, None

    val_rows = max(1, int(total_rows * val_fraction))
    train_rows = total_rows - val_rows
    if train_rows < 1:
        return dataset, None

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total_rows, generator=generator)
    val_indices = permutation[:val_rows].tolist()
    train_indices = permutation[val_rows:].tolist()
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _evaluate(
    model,
    dataloader,
    criterion,
    target_family,
    device,
    amp_enabled,
    pin_memory,
    epoch_num=0,
    epochs=0,
    current_lr=None,
    progress_every=0,
    progress_seconds=0.0,
    progress_callback=None,
    prompt_loss_weight=1.0,
    pooled_loss_weight=1.0,
    sdxl_loss_config=None,
):
    if dataloader is None:
        return None

    model.eval()
    metric_sums = _init_metric_sums(target_family, device)
    steps = 0
    rows_done = 0
    pass_start = perf_counter()
    last_emit_s = pass_start
    total_steps = len(dataloader)
    total_rows = len(dataloader.dataset)

    with torch.inference_mode():
        for batch_qwen, batch_target in dataloader:
            batch_qwen = batch_qwen.to(device, non_blocking=pin_memory)
            batch_target = _move_target_bundle_to_device(
                batch_target, device, pin_memory
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                predictions = model(batch_qwen)
                _, batch_metrics = _compute_loss_metrics(
                    predictions,
                    batch_target,
                    target_family,
                    criterion,
                    prompt_loss_weight=prompt_loss_weight,
                    pooled_loss_weight=pooled_loss_weight,
                    sdxl_loss_config=sdxl_loss_config,
                )

            _accumulate_metric_sums(metric_sums, batch_metrics)
            steps += 1
            rows_done += _bundle_batch_size(batch_target)

            if _should_emit_progress(
                steps, last_emit_s, progress_every, progress_seconds
            ):
                elapsed_s = max(perf_counter() - pass_start, 1e-6)
                avg_metrics = _finalize_metric_sums(metric_sums, steps)
                _emit_progress(
                    progress_callback,
                    {
                        "split": "val",
                        "epoch": epoch_num,
                        "epochs": epochs,
                        "step": steps,
                        "total_steps": total_steps,
                        "rows": rows_done,
                        "total_rows": total_rows,
                        **avg_metrics,
                        "rows_per_s": rows_done / elapsed_s,
                        "lr": current_lr,
                    },
                )
                last_emit_s = perf_counter()

    model.train()
    return _finalize_metric_sums(metric_sums, steps)


def _save_best_checkpoint(
    model,
    path,
    epoch,
    best_metric_name,
    best_metric,
    best_mse,
    input_dim,
    target_dim,
    hidden_dim,
):
    torch.save(
        {
            "epoch": epoch,
            "best_metric_name": best_metric_name,
            "best_metric": best_metric,
            "best_mse": best_mse,
            "input_dim": input_dim,
            "target_dim": target_dim,
            "hidden_dim": hidden_dim,
            "model_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            },
        },
        path,
    )


def _restore_best_checkpoint(model, path, device):
    if not os.path.exists(path):
        return None
    best_checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    return best_checkpoint


def _run_archive_pass(
    reader,
    model,
    criterion,
    target_family,
    device,
    amp_enabled,
    pin_memory,
    batch_size,
    val_fraction,
    split_seed,
    split,
    optimizer=None,
    scaler=None,
    scheduler=None,
    grad_clip=0.0,
    max_steps_remaining=0,
    epoch_num=0,
    epochs=0,
    total_rows=0,
    total_steps=0,
    progress_every=0,
    progress_seconds=0.0,
    progress_callback=None,
    prompt_loss_weight=1.0,
    pooled_loss_weight=1.0,
    sdxl_loss_config=None,
    timing_enabled=False,
):
    is_train = split == "train"
    if is_train:
        model.train()
    else:
        model.eval()

    metric_sums = _init_metric_sums(target_family, device)
    steps = 0
    used_steps = 0
    rows_done = 0
    pass_start = perf_counter()
    last_emit_s = pass_start
    instrumentation = None
    if timing_enabled:
        instrumentation = {
            "batch_count": 0,
            "batch_rows": 0.0,
            "chunk_count": 0,
            "chunk_rows": 0.0,
            "chunk_wait_s": 0.0,
            "copy_s": 0.0,
            "compute_s": 0.0,
        }

    def _run_batch(batch_qwen, batch_target):
        nonlocal steps, used_steps, rows_done, last_emit_s

        batch_rows = _bundle_batch_size(batch_target)
        if instrumentation is not None:
            instrumentation["batch_count"] += 1
            instrumentation["batch_rows"] += batch_rows

        copy_start_event, copy_end_event, copy_start_s = _start_timing_window(
            device, timing_enabled
        )

        batch_qwen = _materialize_qwen_batch(batch_qwen)
        if pin_memory:
            batch_qwen = _pin_qwen_bundle(batch_qwen)
            batch_target = _pin_target_bundle(batch_target)
        batch_qwen = _move_qwen_bundle_to_device(batch_qwen, device, pin_memory)
        batch_target = _move_target_bundle_to_device(batch_target, device, pin_memory)
        if instrumentation is not None:
            instrumentation["copy_s"] += _finish_timing_window(
                copy_start_event, copy_end_event, copy_start_s
            )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        compute_start_event, compute_end_event, compute_start_s = _start_timing_window(
            device, timing_enabled
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            predictions = _forward_qwen_model(model, batch_qwen)
            loss, batch_metrics = _compute_loss_metrics(
                predictions,
                batch_target,
                target_family,
                criterion,
                prompt_loss_weight=prompt_loss_weight,
                pooled_loss_weight=pooled_loss_weight,
                sdxl_loss_config=sdxl_loss_config,
            )

        if is_train:
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            used_steps += 1
        if instrumentation is not None:
            instrumentation["compute_s"] += _finish_timing_window(
                compute_start_event, compute_end_event, compute_start_s
            )

        _accumulate_metric_sums(metric_sums, batch_metrics)
        steps += 1
        rows_done += batch_rows

        if _should_emit_progress(steps, last_emit_s, progress_every, progress_seconds):
            elapsed_s = max(perf_counter() - pass_start, 1e-6)
            current_lr = None
            if optimizer is not None:
                current_lr = optimizer.param_groups[0]["lr"]
            avg_metrics = _finalize_metric_sums(metric_sums, steps)
            avg_metrics.update(_finalize_stream_instrumentation(instrumentation))
            _emit_progress(
                progress_callback,
                {
                    "split": split,
                    "epoch": epoch_num,
                    "epochs": epochs,
                    "step": steps,
                    "total_steps": total_steps,
                    "rows": rows_done,
                    "total_rows": total_rows,
                    **avg_metrics,
                    "rows_per_s": rows_done / elapsed_s,
                    "lr": current_lr,
                },
            )
            last_emit_s = perf_counter()

        return (
            is_train and max_steps_remaining > 0 and used_steps >= max_steps_remaining
        )

    context = torch.enable_grad() if is_train else torch.inference_mode()
    pending_qwen = None
    pending_target = None
    chunk_iterator = iter(reader.iter_chunks())
    with context:
        while True:
            chunk_wait_start = perf_counter()
            try:
                indices, q_chunk_np, target_chunk_np = next(chunk_iterator)
            except StopIteration:
                break
            if instrumentation is not None:
                instrumentation["chunk_count"] += 1
                instrumentation["chunk_rows"] += float(len(indices))
                instrumentation["chunk_wait_s"] += perf_counter() - chunk_wait_start

            val_mask = _split_mask_from_indices(indices, val_fraction, split_seed)
            local_mask = ~val_mask if is_train else val_mask
            local_count = int(local_mask.sum())
            if local_count == 0:
                continue

            q_chunk = _qwen_bundle_from_numpy(q_chunk_np)
            target_chunk = _bundle_from_numpy(target_chunk_np)
            local_indices = torch.from_numpy(
                np.flatnonzero(local_mask).astype(np.int64, copy=False)
            )

            if is_train:
                order = local_indices[torch.randperm(local_indices.numel())]
            else:
                order = local_indices

            q_local = _select_qwen_bundle(q_chunk, order)
            target_local = _select_target_bundle(target_chunk, order)
            if pending_qwen is not None:
                q_local = _concat_qwen_bundles(pending_qwen, q_local)
                target_local = _concat_target_bundles(pending_target, target_local)
                pending_qwen = None
                pending_target = None

            q_local_rows = _qwen_rows(q_local)
            limit = (q_local_rows // batch_size) * batch_size
            stop_requested = False
            for start in range(0, limit, batch_size):
                stop_requested = _run_batch(
                    _slice_qwen_bundle(q_local, start, start + batch_size),
                    _slice_target_bundle(target_local, start, start + batch_size),
                )
                if stop_requested:
                    break

            if stop_requested:
                del q_chunk, target_chunk, local_indices, order, q_local, target_local
                break

            if limit < q_local_rows:
                pending_qwen = _slice_qwen_bundle(q_local, limit, q_local_rows)
                pending_target = {
                    key: value[limit:].clone() for key, value in target_local.items()
                }

            del q_chunk, target_chunk, local_indices, order, q_local, target_local

        if (
            pending_qwen is not None
            and _qwen_rows(pending_qwen) > 0
            and (
                not is_train
                or max_steps_remaining <= 0
                or used_steps < max_steps_remaining
            )
        ):
            _run_batch(pending_qwen, pending_target)

    metrics = _finalize_metric_sums(metric_sums, steps)
    metrics.update(_finalize_stream_instrumentation(instrumentation))
    return metrics, used_steps


# ==========================================
# 3. THE TRAINING LOOP
# ==========================================
def train_projector(
    parquet_path="embedded_chunks/checkpoint_latest.parquet",
    epochs=None,
    batch_size=None,
    max_steps=None,
    use_archives=True,
    archive_root=OUTPUT_DIR,
    qwen_archive_root=QWEN_OUTPUT_DIR,
    max_samples=None,
    progress_callback=None,
    target_family=None,
):
    target_family = get_target_family(
        target_family
        or os.getenv("TRAIN_TARGET_FAMILY", os.getenv("TARGET_FAMILY", "sd"))
    )
    target_layout = get_target_layout(archive_root, target_family)
    ensure_target_root(target_layout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = _env_int("TRAIN_EPOCHS", 50) if epochs is None else epochs
    batch_size = _env_int("TRAIN_BATCH_SIZE", 512) if batch_size is None else batch_size
    max_steps = _env_int("TRAIN_MAX_STEPS", 0) if max_steps is None else max_steps
    max_samples = (
        _env_int("TRAIN_MAX_SAMPLES", 0) if max_samples is None else max_samples
    )
    max_samples = None if max_samples <= 0 else max_samples
    archive_in_memory_limit = _env_int("TRAIN_ARCHIVE_IN_MEMORY_LIMIT", 100_000)
    archive_threads = _env_int("TRAIN_ARCHIVE_THREADS", 2)
    hidden_dim = _env_int("TRAIN_HIDDEN_DIM", 4096)
    sdxl_prompt_token_dim = _env_int("TRAIN_SDXL_PROMPT_TOKEN_DIM", 256)
    sdxl_trunk_depth = _env_int("TRAIN_SDXL_TRUNK_DEPTH", 1)
    sdxl_residual_trunk = _env_int("TRAIN_SDXL_RESIDUAL_TRUNK", 1) > 0
    sdxl_prompt_head_hidden_dim = _env_int("TRAIN_SDXL_PROMPT_HEAD_DIM", 512)
    sdxl_pooled_head_hidden_dim = _env_int("TRAIN_SDXL_POOLED_HEAD_DIM", 2048)
    sdxl_projector_arch = _env_str("TRAIN_SDXL_PROJECTOR_ARCH", "mlp")
    sdxl_resampler_depth = _env_int("TRAIN_SDXL_RESAMPLER_DEPTH", 2)
    sdxl_resampler_heads = _env_int("TRAIN_SDXL_RESAMPLER_HEADS", 8)
    sdxl_resampler_ff_mult = _env_int("TRAIN_SDXL_RESAMPLER_FF_MULT", 4)
    sdxl_resampler_pooled_queries = _env_int("TRAIN_SDXL_RESAMPLER_POOLED_QUERIES", 1)
    sdxl_use_output_calibrator = _env_bool("TRAIN_SDXL_USE_OUTPUT_CALIBRATOR", False)
    val_fraction = _env_float("TRAIN_VAL_SPLIT", 0.1)
    split_seed = _env_int("TRAIN_SPLIT_SEED", 1337)
    max_lr = _env_float("TRAIN_MAX_LR", 1e-4)
    weight_decay = _env_float("TRAIN_WEIGHT_DECAY", 1e-5)
    pct_start = _env_float("TRAIN_PCT_START", 0.1)
    grad_clip = _env_float("TRAIN_GRAD_CLIP", 1.0)
    progress_every = _env_int("TRAIN_PROGRESS_EVERY", 50)
    progress_seconds = _env_float("TRAIN_PROGRESS_SECONDS", 30.0)
    timing_enabled = _env_int("TRAIN_TIMING", 0) > 0
    prompt_loss_weight = _env_float("TRAIN_SDXL_PROMPT_LOSS_WEIGHT", 1.0)
    pooled_loss_weight = _env_float("TRAIN_SDXL_POOLED_LOSS_WEIGHT", 1.0)
    sdxl_loss_config = _build_sdxl_loss_config() if target_family == "sdxl" else None
    sdxl_monitor_config = (
        _build_sdxl_monitor_config() if target_family == "sdxl" else None
    )
    best_checkpoint_path = os.getenv(
        "TRAIN_BEST_CHECKPOINT", target_layout.best_checkpoint_path
    )
    num_workers = _env_int("TRAIN_NUM_WORKERS", 0)
    pin_memory = device.type == "cuda"
    amp_enabled = device.type == "cuda"

    print(f"Training on device: {device}")
    print(f"Target family: {target_family}")
    _log_gpu_runtime(device)

    qwen_has_sequence_inputs = use_archives and _qwen_archives_have_sequence_inputs(
        os.path.join(qwen_archive_root, "archive")
    )
    if target_family == "sdxl":
        sdxl_projector_arch = _resolve_sdxl_projector_architecture(
            sdxl_projector_arch,
            qwen_has_sequence_inputs,
        )
    stream_archives = use_archives and (
        qwen_has_sequence_inputs
        or max_samples is None
        or max_samples > archive_in_memory_limit
    )
    if stream_archives:
        reader = ArchiveChunkReader(
            qwen_archive_dir=os.path.join(qwen_archive_root, "archive"),
            clip_archive_dir=os.path.join(archive_root, "clip_archive"),
            target_layout=target_layout,
            max_samples=max_samples,
            archive_threads=archive_threads,
        )
        split_summary = reader.summarize(val_fraction, split_seed, batch_size)
        if target_family == "sdxl":
            model = _build_sdxl_projector(
                sdxl_projector_arch,
                qwen_dim=reader.input_dim,
                prompt_seq_len=reader.prompt_seq_len,
                prompt_dim=reader.prompt_dim,
                pooled_dim=reader.pooled_dim,
                hidden_dim=hidden_dim,
                prompt_token_dim=sdxl_prompt_token_dim,
                trunk_depth=sdxl_trunk_depth,
                residual_trunk=sdxl_residual_trunk,
                prompt_head_hidden_dim=sdxl_prompt_head_hidden_dim,
                pooled_head_hidden_dim=sdxl_pooled_head_hidden_dim,
                resampler_depth=sdxl_resampler_depth,
                resampler_heads=sdxl_resampler_heads,
                resampler_ff_mult=sdxl_resampler_ff_mult,
                pooled_query_count=sdxl_resampler_pooled_queries,
                has_sequence_inputs=reader.has_sequence_inputs,
                use_output_calibrator=sdxl_use_output_calibrator,
            ).to(device)
        else:
            model = QwenToSDProjector(
                qwen_dim=reader.input_dim,
                sd_dim=reader.target_dim,
                hidden_dim=hidden_dim,
            ).to(device)
        criterion = _build_pointwise_criterion(target_family, sdxl_loss_config)
        optimizer = optim.AdamW(
            model.parameters(), lr=max_lr, weight_decay=weight_decay
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        total_steps = epochs * split_summary["train_steps"]
        if max_steps > 0:
            total_steps = min(total_steps, max_steps)
        scheduler = None
        if total_steps > 1:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                div_factor=25.0,
                final_div_factor=1e4,
            )

        print(
            f"Starting streaming training on {split_summary['train_rows']:,} train samples"
            + (
                f" + {split_summary['val_rows']:,} val samples"
                if split_summary["val_rows"] > 0
                else ""
            )
            + f" | batch_size={batch_size} | epochs={epochs} | amp={amp_enabled} | archive_threads={archive_threads}"
        )
        if reader.has_sequence_inputs:
            print(
                "Qwen archive format: token_sequence (current projector path uses stored pooled source vectors until a sequence-aware projector is selected)."
            )
        if target_family == "sdxl":
            print(f"SDXL projector architecture: {sdxl_projector_arch}")
        print(
            f"Optimizer=AdamW(max_lr={max_lr}, weight_decay={weight_decay}) | "
            f"hidden_dim={hidden_dim} | val_split={val_fraction:.2f} | grad_clip={grad_clip}"
            + (
                f" | prompt_w={prompt_loss_weight} | pooled_w={pooled_loss_weight}"
                if target_family == "sdxl"
                else ""
            )
        )
        if sdxl_loss_config is not None:
            print(
                "SDXL loss config: "
                f"pointwise={sdxl_loss_config.pointwise_name} "
                f"(delta={sdxl_loss_config.huber_delta:.3f}) | "
                f"prompt_cos_w={sdxl_loss_config.prompt_cosine_weight:.3f} | "
                f"pooled_cos_w={sdxl_loss_config.pooled_cosine_weight:.3f} | "
                f"prompt_norm_w={sdxl_loss_config.prompt_norm_weight:.3f} | "
                f"pooled_norm_w={sdxl_loss_config.pooled_norm_weight:.3f}"
            )
            print(
                "SDXL checkpoint monitor: "
                f"metric={sdxl_monitor_config.metric_name} | "
                f"norm_w={sdxl_monitor_config.norm_ratio_weight:.3f} | "
                f"std_w={sdxl_monitor_config.std_ratio_weight:.3f}"
            )
            if sdxl_projector_arch == "resampler":
                print(
                    "SDXL resampler architecture: "
                    f"depth={sdxl_resampler_depth} | "
                    f"heads={sdxl_resampler_heads} | "
                    f"ff_mult={sdxl_resampler_ff_mult} | "
                    f"pooled_queries={sdxl_resampler_pooled_queries}"
                )
            else:
                print(
                    "SDXL architecture: "
                    f"prompt_token_dim={sdxl_prompt_token_dim} | "
                    f"trunk_depth={sdxl_trunk_depth} | "
                    f"residual_trunk={sdxl_residual_trunk} | "
                    f"prompt_head_dim={sdxl_prompt_head_hidden_dim} | "
                    f"pooled_head_dim={sdxl_pooled_head_hidden_dim}"
                )
        print(
            f"Progress updates every {progress_every} batch(es) or {progress_seconds:.0f}s.",
            flush=True,
        )
        if timing_enabled:
            print(
                "Timing instrumentation enabled: batch_rows, chunk_rows, chunk_wait_ms, copy_ms, gpu_ms.",
                flush=True,
            )

        global_step = 0
        best_metric = float("inf")
        best_epoch = 0
        for epoch in range(epochs):
            steps_left = max(0, max_steps - global_step) if max_steps > 0 else 0
            train_metrics, used_steps = _run_archive_pass(
                reader,
                model,
                criterion,
                target_family,
                device,
                amp_enabled,
                pin_memory,
                batch_size,
                val_fraction,
                split_seed,
                "train",
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                grad_clip=grad_clip,
                max_steps_remaining=steps_left,
                epoch_num=epoch + 1,
                epochs=epochs,
                total_rows=split_summary["train_rows"],
                total_steps=split_summary["train_steps"],
                progress_every=progress_every,
                progress_seconds=progress_seconds,
                progress_callback=progress_callback,
                prompt_loss_weight=prompt_loss_weight,
                pooled_loss_weight=pooled_loss_weight,
                sdxl_loss_config=sdxl_loss_config,
                timing_enabled=timing_enabled,
            )
            global_step += used_steps

            val_metrics = None
            if split_summary["val_rows"] > 0:
                val_metrics, _ = _run_archive_pass(
                    reader,
                    model,
                    criterion,
                    target_family,
                    device,
                    amp_enabled,
                    pin_memory,
                    batch_size,
                    val_fraction,
                    split_seed,
                    "val",
                    epoch_num=epoch + 1,
                    epochs=epochs,
                    total_rows=split_summary["val_rows"],
                    total_steps=split_summary["val_steps"],
                    progress_every=progress_every,
                    progress_seconds=progress_seconds,
                    progress_callback=progress_callback,
                    prompt_loss_weight=prompt_loss_weight,
                    pooled_loss_weight=pooled_loss_weight,
                    sdxl_loss_config=sdxl_loss_config,
                    timing_enabled=timing_enabled,
                )

            current_lr = optimizer.param_groups[0]["lr"]
            monitor_source_metrics = (
                train_metrics if val_metrics is None else val_metrics
            )
            monitor_value, monitor_name = _compute_monitor_metric(
                target_family,
                monitor_source_metrics,
                sdxl_monitor_config,
            )
            monitor_source_metrics["monitor"] = monitor_value
            monitor_source_metrics["monitor_name"] = monitor_name
            if monitor_value < best_metric:
                best_metric = monitor_value
                best_epoch = epoch + 1
                _save_best_checkpoint(
                    model,
                    best_checkpoint_path,
                    best_epoch,
                    monitor_name,
                    best_metric,
                    monitor_source_metrics["mse"],
                    reader.input_dim,
                    reader.target_dim,
                    hidden_dim,
                )

            should_log = (
                (epoch + 1) % 10 == 0
                or epoch == 0
                or (max_steps > 0 and global_step >= max_steps)
            )
            if should_log:
                log_line = f"Epoch {epoch+1}/{epochs} | " + _format_epoch_metrics(
                    "train_", train_metrics
                )
                if val_metrics is not None:
                    log_line += " | " + _format_epoch_metrics("val_", val_metrics)
                log_line += f" | lr={current_lr:.6e}"
                print(log_line)

            if max_steps > 0 and global_step >= max_steps:
                break

        best_checkpoint = _restore_best_checkpoint(model, best_checkpoint_path, device)
        if best_checkpoint is not None:
            best_metric_name = best_checkpoint.get("best_metric_name", "mse")
            best_metric_value = best_checkpoint.get(
                "best_metric", best_checkpoint["best_mse"]
            )
            print(
                f"Restored best checkpoint from epoch {best_checkpoint['epoch']} "
                f"with {best_metric_name}={best_metric_value:.6f}"
                + (
                    ""
                    if best_metric_name == "mse"
                    else f" | mse={best_checkpoint['best_mse']:.6f}"
                )
            )

        print("Training complete!")
        return model

    if use_archives:
        dataset = ArchiveEmbeddingDataset(
            qwen_archive_dir=os.path.join(qwen_archive_root, "archive"),
            clip_archive_dir=os.path.join(archive_root, "clip_archive"),
            target_layout=target_layout,
            max_samples=max_samples,
        )
    else:
        if target_family != "sd":
            raise NotImplementedError(
                "Non-archive SDXL training is not wired yet. Use archive-backed training for SDXL targets."
            )
        dataset = ParquetEmbeddingDataset(parquet_path, max_samples=max_samples)
    train_dataset, val_dataset = _split_dataset(dataset, val_fraction, split_seed)
    if target_family == "sdxl":
        model = _build_sdxl_projector(
            sdxl_projector_arch,
            qwen_dim=dataset.input_dim,
            prompt_seq_len=dataset.prompt_seq_len,
            prompt_dim=dataset.prompt_dim,
            pooled_dim=dataset.pooled_dim,
            hidden_dim=hidden_dim,
            prompt_token_dim=sdxl_prompt_token_dim,
            trunk_depth=sdxl_trunk_depth,
            residual_trunk=sdxl_residual_trunk,
            prompt_head_hidden_dim=sdxl_prompt_head_hidden_dim,
            pooled_head_hidden_dim=sdxl_pooled_head_hidden_dim,
            resampler_depth=sdxl_resampler_depth,
            resampler_heads=sdxl_resampler_heads,
            resampler_ff_mult=sdxl_resampler_ff_mult,
            pooled_query_count=sdxl_resampler_pooled_queries,
            has_sequence_inputs=False,
            use_output_calibrator=sdxl_use_output_calibrator,
        ).to(device)
    else:
        model = QwenToSDProjector(
            qwen_dim=dataset.input_dim,
            sd_dim=dataset.target_dim,
            hidden_dim=hidden_dim,
        ).to(device)
    criterion = _build_pointwise_criterion(target_family, sdxl_loss_config)
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

    total_steps = epochs * len(train_dataloader)
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    scheduler = None
    if total_steps > 1:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=25.0,
            final_div_factor=1e4,
        )

    print(
        f"Starting training on {len(train_dataset):,} train samples"
        + (f" + {len(val_dataset):,} val samples" if val_dataset is not None else "")
        + f" | batch_size={batch_size} | epochs={epochs} | amp={amp_enabled}"
    )
    if target_family == "sdxl":
        print(f"SDXL projector architecture: {sdxl_projector_arch}")
    print(
        f"Optimizer=AdamW(max_lr={max_lr}, weight_decay={weight_decay}) | "
        f"hidden_dim={hidden_dim} | val_split={val_fraction:.2f} | grad_clip={grad_clip}"
        + (
            f" | prompt_w={prompt_loss_weight} | pooled_w={pooled_loss_weight}"
            if target_family == "sdxl"
            else ""
        )
    )
    if sdxl_loss_config is not None:
        print(
            "SDXL loss config: "
            f"pointwise={sdxl_loss_config.pointwise_name} "
            f"(delta={sdxl_loss_config.huber_delta:.3f}) | "
            f"prompt_cos_w={sdxl_loss_config.prompt_cosine_weight:.3f} | "
            f"pooled_cos_w={sdxl_loss_config.pooled_cosine_weight:.3f} | "
            f"prompt_norm_w={sdxl_loss_config.prompt_norm_weight:.3f} | "
            f"pooled_norm_w={sdxl_loss_config.pooled_norm_weight:.3f}"
        )
        print(
            "SDXL checkpoint monitor: "
            f"metric={sdxl_monitor_config.metric_name} | "
            f"norm_w={sdxl_monitor_config.norm_ratio_weight:.3f} | "
            f"std_w={sdxl_monitor_config.std_ratio_weight:.3f}"
        )
        if sdxl_projector_arch == "resampler":
            print(
                "SDXL resampler architecture: "
                f"depth={sdxl_resampler_depth} | "
                f"heads={sdxl_resampler_heads} | "
                f"ff_mult={sdxl_resampler_ff_mult} | "
                f"pooled_queries={sdxl_resampler_pooled_queries}"
            )
        else:
            print(
                "SDXL architecture: "
                f"prompt_token_dim={sdxl_prompt_token_dim} | "
                f"trunk_depth={sdxl_trunk_depth} | "
                f"residual_trunk={sdxl_residual_trunk} | "
                f"prompt_head_dim={sdxl_prompt_head_hidden_dim} | "
                f"pooled_head_dim={sdxl_pooled_head_hidden_dim}"
            )
    print(
        f"Progress updates every {progress_every} batch(es) or {progress_seconds:.0f}s.",
        flush=True,
    )
    model.train()
    global_step = 0
    best_metric = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        metric_sums = _init_metric_sums(target_family, device)
        steps_this_epoch = 0
        rows_done = 0
        pass_start = perf_counter()
        last_emit_s = pass_start
        for batch_qwen, batch_target in train_dataloader:
            batch_qwen = batch_qwen.to(device, non_blocking=pin_memory)
            batch_target = _move_target_bundle_to_device(
                batch_target, device, pin_memory
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                predictions = model(batch_qwen)
                loss, batch_metrics = _compute_loss_metrics(
                    predictions,
                    batch_target,
                    target_family,
                    criterion,
                    prompt_loss_weight=prompt_loss_weight,
                    pooled_loss_weight=pooled_loss_weight,
                    sdxl_loss_config=sdxl_loss_config,
                )

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            _accumulate_metric_sums(metric_sums, batch_metrics)
            steps_this_epoch += 1
            global_step += 1
            rows_done += _bundle_batch_size(batch_target)

            if _should_emit_progress(
                steps_this_epoch, last_emit_s, progress_every, progress_seconds
            ):
                elapsed_s = max(perf_counter() - pass_start, 1e-6)
                avg_metrics = _finalize_metric_sums(metric_sums, steps_this_epoch)
                _emit_progress(
                    progress_callback,
                    {
                        "split": "train",
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "step": steps_this_epoch,
                        "total_steps": len(train_dataloader),
                        "rows": rows_done,
                        "total_rows": len(train_dataset),
                        **avg_metrics,
                        "rows_per_s": rows_done / elapsed_s,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                )
                last_emit_s = perf_counter()

            if max_steps > 0 and global_step >= max_steps:
                break

        train_metrics = _finalize_metric_sums(metric_sums, steps_this_epoch)
        val_metrics = _evaluate(
            model,
            val_dataloader,
            criterion,
            target_family,
            device,
            amp_enabled,
            pin_memory,
            epoch_num=epoch + 1,
            epochs=epochs,
            current_lr=optimizer.param_groups[0]["lr"],
            progress_every=progress_every,
            progress_seconds=progress_seconds,
            progress_callback=progress_callback,
            prompt_loss_weight=prompt_loss_weight,
            pooled_loss_weight=pooled_loss_weight,
            sdxl_loss_config=sdxl_loss_config,
        )
        current_lr = optimizer.param_groups[0]["lr"]

        monitor_source_metrics = train_metrics if val_metrics is None else val_metrics
        monitor_value, monitor_name = _compute_monitor_metric(
            target_family,
            monitor_source_metrics,
            sdxl_monitor_config,
        )
        monitor_source_metrics["monitor"] = monitor_value
        monitor_source_metrics["monitor_name"] = monitor_name
        if monitor_value < best_metric:
            best_metric = monitor_value
            best_epoch = epoch + 1
            _save_best_checkpoint(
                model,
                best_checkpoint_path,
                best_epoch,
                monitor_name,
                best_metric,
                monitor_source_metrics["mse"],
                dataset.input_dim,
                dataset.target_dim,
                hidden_dim,
            )

        should_log = (
            (epoch + 1) % 10 == 0
            or epoch == 0
            or (max_steps > 0 and global_step >= max_steps)
        )
        if should_log:
            log_line = f"Epoch {epoch+1}/{epochs} | " + _format_epoch_metrics(
                "train_", train_metrics
            )
            if val_metrics is not None:
                log_line += " | " + _format_epoch_metrics("val_", val_metrics)
            log_line += f" | lr={current_lr:.6e}"
            print(log_line)

        if max_steps > 0 and global_step >= max_steps:
            break

    best_checkpoint = _restore_best_checkpoint(model, best_checkpoint_path, device)
    if best_checkpoint is not None:
        best_metric_name = best_checkpoint.get("best_metric_name", "mse")
        best_metric_value = best_checkpoint.get(
            "best_metric", best_checkpoint["best_mse"]
        )
        print(
            f"Restored best checkpoint from epoch {best_checkpoint['epoch']} "
            f"with {best_metric_name}={best_metric_value:.6f}"
            + (
                ""
                if best_metric_name == "mse"
                else f" | mse={best_checkpoint['best_mse']:.6f}"
            )
        )

    print("Training complete!")
    return model


# ==========================================
# 4. GGUF EXPORT FOR LLAMA.CPP
# ==========================================
def _gguf_field_value(reader, name, default=None):
    field = reader.fields.get(name)
    if field is None:
        return default
    value = field.contents()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_projector_from_gguf(gguf_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    reader = gguf.GGUFReader(gguf_path)
    target_family = _gguf_field_value(reader, "projector.target_family", "sd")
    tensor_map = {
        tensor.name: torch.from_numpy(np.array(tensor.data, copy=True)).float()
        for tensor in reader.tensors
    }

    if target_family == "sdxl":
        state_dict_tensors = {
            name[len("state_dict.") :]: tensor
            for name, tensor in tensor_map.items()
            if name.startswith("state_dict.")
        }
        if state_dict_tensors:
            architecture = _gguf_field_value(reader, "projector.architecture", None)
            if architecture is None:
                architecture = (
                    "resampler" if "latent_queries" in state_dict_tensors else "mlp"
                )

            if architecture == "resampler":
                input_projection_w = state_dict_tensors["input_projection.0.weight"]
                prompt_head_w = state_dict_tensors["prompt_head.3.weight"]
                pooled_head_w = state_dict_tensors["pooled_head.3.weight"]
                latent_queries = state_dict_tensors["latent_queries"]

                qwen_dim = int(
                    _gguf_field_value(
                        reader, "projector.qwen_dim", input_projection_w.shape[1]
                    )
                )
                hidden_dim = int(
                    _gguf_field_value(
                        reader, "projector.hidden_dim", input_projection_w.shape[0]
                    )
                )
                prompt_seq_len = int(
                    _gguf_field_value(reader, "projector.prompt_seq_len", 77)
                )
                pooled_query_count = int(
                    _gguf_field_value(
                        reader,
                        "projector.pooled_query_count",
                        max(latent_queries.shape[1] - prompt_seq_len, 1),
                    )
                )
                prompt_dim = int(
                    _gguf_field_value(
                        reader, "projector.prompt_dim", prompt_head_w.shape[0]
                    )
                )
                pooled_dim = int(
                    _gguf_field_value(
                        reader, "projector.pooled_dim", pooled_head_w.shape[0]
                    )
                )
                resampler_depth = int(
                    _gguf_field_value(
                        reader,
                        "projector.resampler_depth",
                        sum(
                            1
                            for name in state_dict_tensors
                            if name.startswith("resampler_blocks.")
                            and name.endswith("cross_attn.in_proj_weight")
                        ),
                    )
                )
                resampler_heads = int(
                    _gguf_field_value(reader, "projector.resampler_heads", 8)
                )
                resampler_ff_dim = int(
                    _gguf_field_value(
                        reader,
                        "projector.resampler_ff_dim",
                        hidden_dim
                        * int(
                            _gguf_field_value(reader, "projector.resampler_ff_mult", 4)
                        ),
                    )
                )
                prompt_head_hidden_dim = int(
                    _gguf_field_value(
                        reader,
                        "projector.prompt_head_hidden_dim",
                        state_dict_tensors["prompt_head.1.weight"].shape[0],
                    )
                )
                pooled_head_hidden_dim = int(
                    _gguf_field_value(
                        reader,
                        "projector.pooled_head_hidden_dim",
                        state_dict_tensors["pooled_head.1.weight"].shape[0],
                    )
                )

                model = QwenTokenToSDXLResampler(
                    qwen_dim=qwen_dim,
                    prompt_seq_len=prompt_seq_len,
                    prompt_dim=prompt_dim,
                    pooled_dim=pooled_dim,
                    hidden_dim=hidden_dim,
                    resampler_depth=resampler_depth,
                    resampler_heads=resampler_heads,
                    resampler_ff_dim=resampler_ff_dim,
                    pooled_query_count=pooled_query_count,
                    prompt_head_hidden_dim=prompt_head_hidden_dim,
                    pooled_head_hidden_dim=pooled_head_hidden_dim,
                )
                model.load_state_dict(state_dict_tensors)
                metadata = {
                    "target_family": target_family,
                    "architecture": architecture,
                    "qwen_dim": qwen_dim,
                    "hidden_dim": hidden_dim,
                    "prompt_seq_len": prompt_seq_len,
                    "prompt_dim": prompt_dim,
                    "pooled_dim": pooled_dim,
                    "resampler_depth": resampler_depth,
                    "resampler_heads": resampler_heads,
                    "resampler_ff_dim": resampler_ff_dim,
                    "resampler_ff_mult": resampler_ff_dim // max(hidden_dim, 1),
                    "pooled_query_count": pooled_query_count,
                    "prompt_head_hidden_dim": prompt_head_hidden_dim,
                    "pooled_head_hidden_dim": pooled_head_hidden_dim,
                }
            else:
                input_projection_w = state_dict_tensors["input_projection.0.weight"]
                prompt_seed_w = state_dict_tensors["prompt_seed.weight"]
                prompt_head_w1 = state_dict_tensors["prompt_projection.1.weight"]
                prompt_head_w2 = state_dict_tensors["prompt_projection.3.weight"]
                pooled_head_w1 = state_dict_tensors["pooled_head.0.weight"]
                pooled_head_w2 = state_dict_tensors["pooled_head.2.weight"]

                qwen_dim = int(
                    _gguf_field_value(
                        reader, "projector.qwen_dim", input_projection_w.shape[1]
                    )
                )
                hidden_dim = int(
                    _gguf_field_value(
                        reader, "projector.hidden_dim", input_projection_w.shape[0]
                    )
                )
                prompt_dim = int(
                    _gguf_field_value(
                        reader, "projector.prompt_dim", prompt_head_w2.shape[0]
                    )
                )
                pooled_dim = int(
                    _gguf_field_value(
                        reader, "projector.pooled_dim", pooled_head_w2.shape[0]
                    )
                )
                prompt_token_dim = int(
                    _gguf_field_value(
                        reader,
                        "projector.prompt_token_dim",
                        prompt_head_w1.shape[1],
                    )
                )
                prompt_seq_len = int(
                    _gguf_field_value(
                        reader,
                        "projector.prompt_seq_len",
                        prompt_seed_w.shape[0] // max(prompt_token_dim, 1),
                    )
                )
                trunk_depth = int(
                    _gguf_field_value(
                        reader,
                        "projector.trunk_depth",
                        sum(
                            1
                            for name in state_dict_tensors
                            if name.startswith("trunk_layers.")
                            and name.endswith(".fc1.weight")
                        ),
                    )
                )
                residual_trunk = bool(
                    int(_gguf_field_value(reader, "projector.residual_trunk", 1))
                )
                prompt_head_hidden_dim = int(
                    _gguf_field_value(
                        reader,
                        "projector.prompt_head_hidden_dim",
                        prompt_head_w1.shape[0],
                    )
                )
                pooled_head_hidden_dim = int(
                    _gguf_field_value(
                        reader,
                        "projector.pooled_head_hidden_dim",
                        pooled_head_w1.shape[0],
                    )
                )
                use_output_calibrator = bool(
                    int(
                        _gguf_field_value(
                            reader,
                            "projector.use_output_calibrator",
                            (
                                1
                                if "prompt_output_calibrator.gain" in state_dict_tensors
                                else 0
                            ),
                        )
                    )
                )

                model = QwenToSDXLProjector(
                    qwen_dim=qwen_dim,
                    prompt_seq_len=prompt_seq_len,
                    prompt_dim=prompt_dim,
                    pooled_dim=pooled_dim,
                    hidden_dim=hidden_dim,
                    prompt_token_dim=prompt_token_dim,
                    trunk_depth=trunk_depth,
                    residual_trunk=residual_trunk,
                    prompt_head_hidden_dim=prompt_head_hidden_dim,
                    pooled_head_hidden_dim=pooled_head_hidden_dim,
                    use_output_calibrator=use_output_calibrator,
                )
                model.load_state_dict(state_dict_tensors)
                metadata = {
                    "target_family": target_family,
                    "architecture": "mlp",
                    "qwen_dim": qwen_dim,
                    "hidden_dim": hidden_dim,
                    "prompt_seq_len": prompt_seq_len,
                    "prompt_dim": prompt_dim,
                    "pooled_dim": pooled_dim,
                    "prompt_token_dim": prompt_token_dim,
                    "trunk_depth": trunk_depth,
                    "residual_trunk": residual_trunk,
                    "prompt_head_hidden_dim": prompt_head_hidden_dim,
                    "pooled_head_hidden_dim": pooled_head_hidden_dim,
                    "use_output_calibrator": use_output_calibrator,
                    "output_mode": "calibrated" if use_output_calibrator else "plain",
                }
        else:
            trunk_w1 = tensor_map["trunk_w1"]
            trunk_b1 = tensor_map["trunk_b1"]
            trunk_w2 = tensor_map["trunk_w2"]
            trunk_b2 = tensor_map["trunk_b2"]
            prompt_seed_w = tensor_map["prompt_seed_w"]
            prompt_seed_b = tensor_map["prompt_seed_b"]
            prompt_proj_w = tensor_map["prompt_proj_w"]
            prompt_proj_b = tensor_map["prompt_proj_b"]
            pooled_w = tensor_map["pooled_w"]
            pooled_b = tensor_map["pooled_b"]

            qwen_dim = int(
                _gguf_field_value(reader, "projector.qwen_dim", trunk_w1.shape[1])
            )
            hidden_dim = int(
                _gguf_field_value(reader, "projector.hidden_dim", trunk_w1.shape[0])
            )
            prompt_dim = int(
                _gguf_field_value(
                    reader, "projector.prompt_dim", prompt_proj_w.shape[0]
                )
            )
            pooled_dim = int(
                _gguf_field_value(reader, "projector.pooled_dim", pooled_w.shape[0])
            )
            prompt_token_dim = int(
                _gguf_field_value(
                    reader, "projector.prompt_token_dim", prompt_proj_w.shape[1]
                )
            )
            prompt_seq_len = int(
                _gguf_field_value(
                    reader,
                    "projector.prompt_seq_len",
                    prompt_seed_w.shape[0] // max(prompt_token_dim, 1),
                )
            )

            model = QwenToSDXLProjector(
                qwen_dim=qwen_dim,
                prompt_seq_len=prompt_seq_len,
                prompt_dim=prompt_dim,
                pooled_dim=pooled_dim,
                hidden_dim=hidden_dim,
                prompt_token_dim=prompt_token_dim,
                trunk_depth=0,
                residual_trunk=False,
                prompt_head_hidden_dim=prompt_dim,
                pooled_head_hidden_dim=hidden_dim,
                use_output_calibrator=False,
            )
            model.input_projection[0].weight.data.copy_(trunk_w1)
            model.input_projection[0].bias.data.copy_(trunk_b1)
            model.input_projection[2].weight.data.copy_(trunk_w2)
            model.input_projection[2].bias.data.copy_(trunk_b2)
            model.prompt_seed.weight.data.copy_(prompt_seed_w)
            model.prompt_seed.bias.data.copy_(prompt_seed_b)
            model.prompt_projection[1].weight.data.copy_(prompt_proj_w)
            model.prompt_projection[1].bias.data.copy_(prompt_proj_b)
            model.prompt_projection[3].weight.data.copy_(
                torch.eye(prompt_dim, dtype=model.prompt_projection[3].weight.dtype)
            )
            model.prompt_projection[3].bias.data.zero_()
            model.pooled_head[0].weight.data.copy_(
                torch.eye(hidden_dim, dtype=model.pooled_head[0].weight.dtype)
            )
            model.pooled_head[0].bias.data.zero_()
            model.pooled_head[2].weight.data.copy_(pooled_w)
            model.pooled_head[2].bias.data.copy_(pooled_b)
            metadata = {
                "target_family": target_family,
                "architecture": "mlp",
                "qwen_dim": qwen_dim,
                "hidden_dim": hidden_dim,
                "prompt_seq_len": prompt_seq_len,
                "prompt_dim": prompt_dim,
                "pooled_dim": pooled_dim,
                "prompt_token_dim": prompt_token_dim,
                "trunk_depth": 0,
                "residual_trunk": False,
                "prompt_head_hidden_dim": prompt_dim,
                "pooled_head_hidden_dim": hidden_dim,
                "use_output_calibrator": False,
                "output_mode": "plain",
            }
    else:
        w1 = tensor_map["proj_w1"]
        b1 = tensor_map["proj_b1"]
        w2 = tensor_map["proj_w2"]
        b2 = tensor_map["proj_b2"]
        qwen_dim = int(_gguf_field_value(reader, "projector.qwen_dim", w1.shape[1]))
        hidden_dim = int(_gguf_field_value(reader, "projector.hidden_dim", w1.shape[0]))
        sd_dim = int(_gguf_field_value(reader, "projector.sd_dim", w2.shape[0]))

        model = QwenToSDProjector(
            qwen_dim=qwen_dim,
            sd_dim=sd_dim,
            hidden_dim=hidden_dim,
        )
        model.load_state_dict(
            {
                "projection.0.weight": w1,
                "projection.0.bias": b1,
                "projection.2.weight": w2,
                "projection.2.bias": b2,
            }
        )
        metadata = {
            "target_family": "sd",
            "qwen_dim": qwen_dim,
            "hidden_dim": hidden_dim,
            "sd_dim": sd_dim,
        }

    model = model.to(device).eval()
    return model, metadata


def export_to_gguf(model, output_filename="qwen_sd_projector.gguf", target_family="sd"):
    target_family = get_target_family(target_family)

    print(f"\nExporting weights to {output_filename}...")

    model.eval()
    model.cpu()
    state_dict = model.state_dict()
    architecture = getattr(model, "architecture_name", "mlp")

    writer = gguf.GGUFWriter(output_filename, "projector")
    schema_version = 2
    if target_family == "sdxl":
        schema_version = 4 if architecture == "resampler" else 3
    writer.add_uint32("projector.schema_version", schema_version)
    writer.add_string("projector.target_family", target_family)

    if target_family == "sdxl":
        writer.add_string("projector.architecture", architecture)
        writer.add_uint32("projector.qwen_dim", int(model.qwen_dim))
        writer.add_uint32("projector.hidden_dim", int(model.hidden_dim))
        writer.add_uint32("projector.prompt_seq_len", int(model.prompt_seq_len))
        writer.add_uint32("projector.prompt_dim", int(model.prompt_dim))
        writer.add_uint32("projector.pooled_dim", int(model.pooled_dim))
        writer.add_string(
            "projector.output_mode",
            "calibrated" if getattr(model, "use_output_calibrator", False) else "plain",
        )
        writer.add_uint32(
            "projector.use_output_calibrator",
            1 if getattr(model, "use_output_calibrator", False) else 0,
        )
        writer.add_uint32(
            "projector.prompt_head_hidden_dim", int(model.prompt_head_hidden_dim)
        )
        writer.add_uint32(
            "projector.pooled_head_hidden_dim", int(model.pooled_head_hidden_dim)
        )
        if architecture == "resampler":
            writer.add_uint32("projector.resampler_depth", int(model.resampler_depth))
            writer.add_uint32("projector.resampler_heads", int(model.resampler_heads))
            writer.add_uint32("projector.resampler_ff_dim", int(model.resampler_ff_dim))
            writer.add_uint32(
                "projector.resampler_ff_mult",
                int(max(model.resampler_ff_dim // max(model.hidden_dim, 1), 1)),
            )
            writer.add_uint32(
                "projector.pooled_query_count", int(model.pooled_query_count)
            )
        else:
            writer.add_uint32("projector.prompt_token_dim", int(model.prompt_token_dim))
            writer.add_uint32("projector.trunk_depth", int(model.trunk_depth))
            writer.add_uint32(
                "projector.residual_trunk", 1 if model.residual_trunk else 0
            )
        for name, tensor in state_dict.items():
            writer.add_tensor(f"state_dict.{name}", tensor.numpy())
    else:
        w1 = state_dict["projection.0.weight"].numpy()
        b1 = state_dict["projection.0.bias"].numpy()
        w2 = state_dict["projection.2.weight"].numpy()
        b2 = state_dict["projection.2.bias"].numpy()

        writer.add_uint32("projector.qwen_dim", int(w1.shape[1]))
        writer.add_uint32("projector.hidden_dim", int(w1.shape[0]))
        writer.add_uint32("projector.sd_dim", int(w2.shape[0]))
        writer.add_tensor("proj_w1", w1)
        writer.add_tensor("proj_b1", b1)
        writer.add_tensor("proj_w2", w2)
        writer.add_tensor("proj_b2", b2)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(
        f"Successfully saved {output_filename}! "
        f"File size: {os.path.getsize(output_filename) / (1024*1024):.2f} MB"
    )


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    target_family = get_target_family(
        os.getenv("TRAIN_TARGET_FAMILY", os.getenv("TARGET_FAMILY", "sd"))
    )
    target_layout = get_target_layout(OUTPUT_DIR, target_family)
    gguf_output_path = os.getenv("TRAIN_GGUF_PATH", target_layout.gguf_path)
    qwen_archives = _list_npz_archives(QWEN_ARCHIVE_DIR, "archive_")
    clip_archives, resolved_target_dir = _resolve_target_archives(target_layout)
    if not qwen_archives or not clip_archives:
        print(
            "Error: Required archives not found. Run runner.py and clip_runner.py first "
            f"to generate Qwen and {target_family} target archive files. "
            f"Expected target archives under {resolved_target_dir}."
        )
        raise SystemExit(1)

    print(f"Using archives: {QWEN_ARCHIVE_DIR} + {resolved_target_dir}")
    trained_model = train_projector(
        use_archives=True,
        archive_root=OUTPUT_DIR,
        qwen_archive_root=QWEN_OUTPUT_DIR,
        target_family=target_family,
    )
    export_to_gguf(
        trained_model,
        gguf_output_path,
        target_family=target_family,
    )
