import os
import math
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


def _bundle_from_numpy(bundle):
    return {key: torch.from_numpy(value) for key, value in bundle.items()}


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
        metric_sums[key] += value


def _finalize_metric_sums(metric_sums, steps):
    return {key: (value / max(1, steps)).item() for key, value in metric_sums.items()}


def _metrics_extra_text(metrics, prefix=""):
    if "prompt_mse" not in metrics:
        return ""
    return (
        f" | {prefix}prompt_mse={metrics['prompt_mse']:.6f}"
        f" | {prefix}pooled_mse={metrics['pooled_mse']:.6f}"
    )


def _format_epoch_metrics(prefix, metrics):
    return (
        f"{prefix}mse={metrics['mse']:.6f} | "
        f"{prefix}mae={metrics['mae']:.6f} | "
        f"{prefix}cos={metrics['cosine']:.6f}"
        f"{_metrics_extra_text(metrics, prefix=prefix)}"
    )


def _compute_loss_metrics(
    predictions,
    target_bundle,
    target_family,
    criterion,
    prompt_loss_weight=1.0,
    pooled_loss_weight=1.0,
):
    if target_family == "sdxl":
        prompt_predictions, pooled_predictions = predictions
        prompt_targets = target_bundle["prompt_embeds"]
        pooled_targets = target_bundle["pooled_prompt_embeds"]

        prompt_loss = criterion(prompt_predictions, prompt_targets)
        pooled_loss = criterion(pooled_predictions, pooled_targets)
        loss = _weighted_average(
            [prompt_loss, pooled_loss],
            [prompt_loss_weight, pooled_loss_weight],
        )

        prompt_predictions_f32 = prompt_predictions.float()
        pooled_predictions_f32 = pooled_predictions.float()
        prompt_targets_f32 = prompt_targets.float()
        pooled_targets_f32 = pooled_targets.float()

        prompt_mae = (prompt_predictions_f32 - prompt_targets_f32).abs().mean().detach()
        pooled_mae = (pooled_predictions_f32 - pooled_targets_f32).abs().mean().detach()
        prompt_cosine = (
            F.cosine_similarity(
                _flatten_for_cosine(prompt_predictions_f32),
                _flatten_for_cosine(prompt_targets_f32),
                dim=-1,
            )
            .mean()
            .detach()
        )
        pooled_cosine = (
            F.cosine_similarity(
                pooled_predictions_f32,
                pooled_targets_f32,
                dim=-1,
            )
            .mean()
            .detach()
        )

        return loss, {
            "mse": _weighted_average(
                [prompt_loss.detach().float(), pooled_loss.detach().float()],
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
            "prompt_mse": prompt_loss.detach().float(),
            "prompt_mae": prompt_mae,
            "prompt_cosine": prompt_cosine,
            "pooled_mse": pooled_loss.detach().float(),
            "pooled_mae": pooled_mae,
            "pooled_cosine": pooled_cosine,
        }

    targets = target_bundle["embedding"]
    loss = criterion(predictions, targets)
    predictions_f32 = predictions.float()
    targets_f32 = targets.float()
    return loss, {
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
        f"{lr_text}{extra_text}",
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


OUTPUT_DIR = "embedded_chunks"
QWEN_ARCHIVE_DIR = os.path.join(OUTPUT_DIR, "archive")
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


class QwenToSDXLProjector(nn.Module):
    def __init__(
        self,
        qwen_dim,
        prompt_seq_len=77,
        prompt_dim=2048,
        pooled_dim=1280,
        hidden_dim=4096,
        prompt_token_dim=256,
    ):
        super().__init__()
        self.prompt_seq_len = prompt_seq_len
        self.prompt_dim = prompt_dim
        self.pooled_dim = pooled_dim
        self.prompt_token_dim = prompt_token_dim

        self.trunk = nn.Sequential(
            nn.Linear(qwen_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prompt_seed = nn.Linear(hidden_dim, prompt_seq_len * prompt_token_dim)
        self.prompt_projection = nn.Sequential(
            nn.GELU(),
            nn.Linear(prompt_token_dim, prompt_dim),
        )
        self.pooled_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, pooled_dim),
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        prompt_seed = self.prompt_seed(trunk_features).view(
            x.shape[0], self.prompt_seq_len, self.prompt_token_dim
        )
        prompt_out = self.prompt_projection(prompt_seed)
        pooled_out = self.pooled_head(trunk_features)
        return prompt_out, pooled_out


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
        for path in qwen_archives:
            data = np.load(path, allow_pickle=False)
            qwen_total += len(data["indices"])
            if qwen_dim is None:
                qwen_dim = int(data["embeddings"].shape[1])

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
    ):
        self.target_family = target_layout.family if target_layout is not None else "sd"
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

        qwen_total = 0
        for path in self.qwen_archives:
            with np.load(path, allow_pickle=False) as data:
                indices = data["indices"].astype(np.int64, copy=False)
                self.qwen_index_arrays.append(indices)
                self.qwen_counts.append(len(indices))
                qwen_total += len(indices)
                if self.input_dim is None:
                    self.input_dim = int(data["embeddings"].shape[1])

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
        train_steps = 0
        val_steps = 0

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
            if chunk_train_rows > 0:
                train_steps += math.ceil(chunk_train_rows / batch_size)
            if chunk_val_rows > 0:
                val_steps += math.ceil(chunk_val_rows / batch_size)

            q_pos += take
            c_pos += take
            rows_left -= take
            if q_pos == len(q_indices):
                q_file_idx += 1
                q_pos = 0
            if c_pos == self.clip_counts[c_file_idx]:
                c_file_idx += 1
                c_pos = 0

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
                "embeddings": data["embeddings"].astype(np.float32, copy=False),
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
                q_current["embeddings"][q_pos : q_pos + take],
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
                    q_current = self._load_qwen_archive(self.qwen_archives[q_file_idx])
            if c_pos == len(c_current["indices"]):
                c_file_idx += 1
                c_pos = 0
                if c_file_idx < len(self.clip_archives):
                    c_current = self._load_target_archive(
                        self.clip_archives[c_file_idx]
                    )


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
    best_mse,
    input_dim,
    target_dim,
    hidden_dim,
):
    torch.save(
        {
            "epoch": epoch,
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

    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for indices, q_chunk_np, target_chunk_np in reader.iter_chunks():
            val_mask = _split_mask_from_indices(indices, val_fraction, split_seed)
            local_mask = ~val_mask if is_train else val_mask
            local_count = int(local_mask.sum())
            if local_count == 0:
                continue

            q_chunk = torch.from_numpy(q_chunk_np)
            target_chunk = _bundle_from_numpy(target_chunk_np)
            local_indices = torch.from_numpy(
                np.flatnonzero(local_mask).astype(np.int64, copy=False)
            )

            if is_train:
                order = local_indices[torch.randperm(local_indices.numel())]
            else:
                order = local_indices

            for start in range(0, order.numel(), batch_size):
                batch_indices = order[start : start + batch_size]
                batch_qwen = q_chunk.index_select(0, batch_indices)
                batch_target = _select_target_bundle(target_chunk, batch_indices)
                if pin_memory:
                    batch_qwen = batch_qwen.pin_memory()
                    batch_target = _pin_target_bundle(batch_target)
                batch_qwen = batch_qwen.to(device, non_blocking=pin_memory)
                batch_target = _move_target_bundle_to_device(
                    batch_target, device, pin_memory
                )

                if is_train:
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
                    )

                if is_train:
                    if amp_enabled:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), grad_clip
                            )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), grad_clip
                            )
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    used_steps += 1

                _accumulate_metric_sums(metric_sums, batch_metrics)
                steps += 1
                rows_done += _bundle_batch_size(batch_target)

                if _should_emit_progress(
                    steps, last_emit_s, progress_every, progress_seconds
                ):
                    elapsed_s = max(perf_counter() - pass_start, 1e-6)
                    current_lr = None
                    if optimizer is not None:
                        current_lr = optimizer.param_groups[0]["lr"]
                    avg_metrics = _finalize_metric_sums(metric_sums, steps)
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

                if (
                    is_train
                    and max_steps_remaining > 0
                    and used_steps >= max_steps_remaining
                ):
                    break

            del q_chunk, target_chunk, local_indices, order
            if (
                is_train
                and max_steps_remaining > 0
                and used_steps >= max_steps_remaining
            ):
                break

    return _finalize_metric_sums(metric_sums, steps), used_steps


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
    hidden_dim = _env_int("TRAIN_HIDDEN_DIM", 4096)
    val_fraction = _env_float("TRAIN_VAL_SPLIT", 0.1)
    split_seed = _env_int("TRAIN_SPLIT_SEED", 1337)
    max_lr = _env_float("TRAIN_MAX_LR", 5e-4)
    weight_decay = _env_float("TRAIN_WEIGHT_DECAY", 1e-5)
    pct_start = _env_float("TRAIN_PCT_START", 0.3)
    grad_clip = _env_float("TRAIN_GRAD_CLIP", 1.0)
    progress_every = _env_int("TRAIN_PROGRESS_EVERY", 50)
    progress_seconds = _env_float("TRAIN_PROGRESS_SECONDS", 30.0)
    prompt_loss_weight = _env_float("TRAIN_SDXL_PROMPT_LOSS_WEIGHT", 1.0)
    pooled_loss_weight = _env_float("TRAIN_SDXL_POOLED_LOSS_WEIGHT", 1.0)
    best_checkpoint_path = os.getenv(
        "TRAIN_BEST_CHECKPOINT", target_layout.best_checkpoint_path
    )
    num_workers = _env_int("TRAIN_NUM_WORKERS", 0)
    pin_memory = device.type == "cuda"
    amp_enabled = device.type == "cuda"

    print(f"Training on device: {device}")
    print(f"Target family: {target_family}")
    if device.type == "cuda":
        print(
            "ROCm launch: "
            f"HSA_OVERRIDE_GFX_VERSION={os.getenv('HSA_OVERRIDE_GFX_VERSION')}, "
            f"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL={os.getenv('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL')}"
        )

    stream_archives = use_archives and (
        max_samples is None or max_samples > archive_in_memory_limit
    )
    if stream_archives:
        reader = ArchiveChunkReader(
            qwen_archive_dir=os.path.join(archive_root, "archive"),
            clip_archive_dir=os.path.join(archive_root, "clip_archive"),
            target_layout=target_layout,
            max_samples=max_samples,
        )
        split_summary = reader.summarize(val_fraction, split_seed, batch_size)
        if target_family == "sdxl":
            model = QwenToSDXLProjector(
                qwen_dim=reader.input_dim,
                prompt_seq_len=reader.prompt_seq_len,
                prompt_dim=reader.prompt_dim,
                pooled_dim=reader.pooled_dim,
                hidden_dim=hidden_dim,
            ).to(device)
        else:
            model = QwenToSDProjector(
                qwen_dim=reader.input_dim,
                sd_dim=reader.target_dim,
                hidden_dim=hidden_dim,
            ).to(device)
        criterion = nn.MSELoss()
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
            + f" | batch_size={batch_size} | epochs={epochs} | amp={amp_enabled}"
        )
        print(
            f"Optimizer=AdamW(max_lr={max_lr}, weight_decay={weight_decay}) | "
            f"hidden_dim={hidden_dim} | val_split={val_fraction:.2f} | grad_clip={grad_clip}"
            + (
                f" | prompt_w={prompt_loss_weight} | pooled_w={pooled_loss_weight}"
                if target_family == "sdxl"
                else ""
            )
        )
        print(
            f"Progress updates every {progress_every} batch(es) or {progress_seconds:.0f}s.",
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
                )

            current_lr = optimizer.param_groups[0]["lr"]
            monitor_mse = (
                train_metrics["mse"] if val_metrics is None else val_metrics["mse"]
            )
            if monitor_mse < best_metric:
                best_metric = monitor_mse
                best_epoch = epoch + 1
                _save_best_checkpoint(
                    model,
                    best_checkpoint_path,
                    best_epoch,
                    best_metric,
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
            print(
                f"Restored best checkpoint from epoch {best_checkpoint['epoch']} "
                f"with mse={best_checkpoint['best_mse']:.6f}"
            )

        print("Training complete!")
        return model

    if use_archives:
        dataset = ArchiveEmbeddingDataset(
            qwen_archive_dir=os.path.join(archive_root, "archive"),
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
        model = QwenToSDXLProjector(
            qwen_dim=dataset.input_dim,
            prompt_seq_len=dataset.prompt_seq_len,
            prompt_dim=dataset.prompt_dim,
            pooled_dim=dataset.pooled_dim,
            hidden_dim=hidden_dim,
        ).to(device)
    else:
        model = QwenToSDProjector(
            qwen_dim=dataset.input_dim,
            sd_dim=dataset.target_dim,
            hidden_dim=hidden_dim,
        ).to(device)
    criterion = nn.MSELoss()
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
    print(
        f"Optimizer=AdamW(max_lr={max_lr}, weight_decay={weight_decay}) | "
        f"hidden_dim={hidden_dim} | val_split={val_fraction:.2f} | grad_clip={grad_clip}"
        + (
            f" | prompt_w={prompt_loss_weight} | pooled_w={pooled_loss_weight}"
            if target_family == "sdxl"
            else ""
        )
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
        )
        current_lr = optimizer.param_groups[0]["lr"]

        monitor_mse = (
            train_metrics["mse"] if val_metrics is None else val_metrics["mse"]
        )
        if monitor_mse < best_metric:
            best_metric = monitor_mse
            best_epoch = epoch + 1
            _save_best_checkpoint(
                model,
                best_checkpoint_path,
                best_epoch,
                best_metric,
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
        print(
            f"Restored best checkpoint from epoch {best_checkpoint['epoch']} "
            f"with mse={best_checkpoint['best_mse']:.6f}"
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
            _gguf_field_value(reader, "projector.prompt_dim", prompt_proj_w.shape[0])
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
        )
        model.load_state_dict(
            {
                "trunk.0.weight": trunk_w1,
                "trunk.0.bias": trunk_b1,
                "trunk.2.weight": trunk_w2,
                "trunk.2.bias": trunk_b2,
                "prompt_seed.weight": prompt_seed_w,
                "prompt_seed.bias": prompt_seed_b,
                "prompt_projection.1.weight": prompt_proj_w,
                "prompt_projection.1.bias": prompt_proj_b,
                "pooled_head.1.weight": pooled_w,
                "pooled_head.1.bias": pooled_b,
            }
        )
        metadata = {
            "target_family": target_family,
            "qwen_dim": qwen_dim,
            "hidden_dim": hidden_dim,
            "prompt_seq_len": prompt_seq_len,
            "prompt_dim": prompt_dim,
            "pooled_dim": pooled_dim,
            "prompt_token_dim": prompt_token_dim,
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

    writer = gguf.GGUFWriter(output_filename, "projector")
    writer.add_uint32("projector.schema_version", 2)
    writer.add_string("projector.target_family", target_family)

    if target_family == "sdxl":
        trunk_w1 = state_dict["trunk.0.weight"].numpy()
        trunk_b1 = state_dict["trunk.0.bias"].numpy()
        trunk_w2 = state_dict["trunk.2.weight"].numpy()
        trunk_b2 = state_dict["trunk.2.bias"].numpy()
        prompt_seed_w = state_dict["prompt_seed.weight"].numpy()
        prompt_seed_b = state_dict["prompt_seed.bias"].numpy()
        prompt_proj_w = state_dict["prompt_projection.1.weight"].numpy()
        prompt_proj_b = state_dict["prompt_projection.1.bias"].numpy()
        pooled_w = state_dict["pooled_head.1.weight"].numpy()
        pooled_b = state_dict["pooled_head.1.bias"].numpy()

        writer.add_uint32("projector.qwen_dim", int(trunk_w1.shape[1]))
        writer.add_uint32("projector.hidden_dim", int(trunk_w1.shape[0]))
        writer.add_uint32("projector.prompt_seq_len", int(model.prompt_seq_len))
        writer.add_uint32("projector.prompt_dim", int(model.prompt_dim))
        writer.add_uint32("projector.pooled_dim", int(model.pooled_dim))
        writer.add_uint32("projector.prompt_token_dim", int(model.prompt_token_dim))
        writer.add_tensor("trunk_w1", trunk_w1)
        writer.add_tensor("trunk_b1", trunk_b1)
        writer.add_tensor("trunk_w2", trunk_w2)
        writer.add_tensor("trunk_b2", trunk_b2)
        writer.add_tensor("prompt_seed_w", prompt_seed_w)
        writer.add_tensor("prompt_seed_b", prompt_seed_b)
        writer.add_tensor("prompt_proj_w", prompt_proj_w)
        writer.add_tensor("prompt_proj_b", prompt_proj_b)
        writer.add_tensor("pooled_w", pooled_w)
        writer.add_tensor("pooled_b", pooled_b)
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
        target_family=target_family,
    )
    export_to_gguf(
        trained_model,
        target_layout.gguf_path,
        target_family=target_family,
    )
