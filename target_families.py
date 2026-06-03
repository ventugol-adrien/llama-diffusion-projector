import json
import os
from dataclasses import asdict, dataclass

SCHEMA_VERSION = 1
DEFAULT_TARGET_FAMILY = "sdxl"
SUPPORTED_TARGET_FAMILIES = ("sd", "sdxl")
DEFAULT_SDXL_OUTPUT_DIR = "/scratch/llama-diffusion-projector"


@dataclass(frozen=True)
class TargetFamilySpec:
    family: str
    display_name: str
    archive_prefix: str
    sequence_length: int | None = None
    prompt_dim: int | None = None
    pooled_dim: int | None = None
    vector_dim: int | None = None
    model_id: str | None = None
    tokenizer_id: str | None = None
    tokenizer_2_id: str | None = None
    text_encoder_id: str | None = None
    text_encoder_2_id: str | None = None


@dataclass(frozen=True)
class TargetLayout:
    family: str
    root_dir: str
    archive_dir: str
    archive_prefix: str
    checkpoint_delta_path: str
    checkpoint_delta_tmp_path: str
    errors_path: str
    errors_tmp_path: str
    manifest_path: str
    best_checkpoint_path: str
    gguf_path: str
    legacy_archive_dir: str | None = None
    legacy_archive_prefix: str | None = None
    legacy_checkpoint_delta_path: str | None = None


TARGET_FAMILY_SPECS = {
    "sd": TargetFamilySpec(
        family="sd",
        display_name="Stable Diffusion (legacy LongCLIP pooled)",
        archive_prefix="target_archive_",
        vector_dim=768,
        model_id="zer0int/LongCLIP-L-Diffusers",
        tokenizer_id="zer0int/LongCLIP-L-Diffusers",
        text_encoder_id="zer0int/LongCLIP-L-Diffusers",
    ),
    "sdxl": TargetFamilySpec(
        family="sdxl",
        display_name="Stable Diffusion XL",
        archive_prefix="target_archive_",
        sequence_length=77,
        prompt_dim=2048,
        pooled_dim=1280,
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        tokenizer_id="stabilityai/stable-diffusion-xl-base-1.0/tokenizer",
        tokenizer_2_id="stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2",
        text_encoder_id="stabilityai/stable-diffusion-xl-base-1.0/text_encoder",
        text_encoder_2_id="stabilityai/stable-diffusion-xl-base-1.0/text_encoder_2",
    ),
}


def normalize_target_family(value):
    family = (value or DEFAULT_TARGET_FAMILY).strip().lower()
    if family not in SUPPORTED_TARGET_FAMILIES:
        supported = ", ".join(SUPPORTED_TARGET_FAMILIES)
        raise ValueError(
            f"Unsupported target family '{family}'. Expected one of: {supported}"
        )
    return family


def get_target_family(explicit=None, env_var="TARGET_FAMILY"):
    return normalize_target_family(
        explicit or os.getenv(env_var, DEFAULT_TARGET_FAMILY)
    )


def get_target_family_spec(family):
    return TARGET_FAMILY_SPECS[normalize_target_family(family)]


def resolve_target_output_dir(output_dir, family):
    family = normalize_target_family(family)
    shared_override = os.getenv("TARGET_OUTPUT_DIR")
    if shared_override:
        return shared_override
    if family == "sdxl":
        return os.getenv("SDXL_TARGET_OUTPUT_DIR", DEFAULT_SDXL_OUTPUT_DIR)
    return output_dir


def get_target_layout(output_dir, family):
    family = normalize_target_family(family)
    spec = get_target_family_spec(family)
    resolved_output_dir = resolve_target_output_dir(output_dir, family)
    root_dir = os.path.join(resolved_output_dir, "targets", family)
    checkpoint_delta_filename = "checkpoint_delta.parquet"
    checkpoint_delta_tmp_filename = "checkpoint_delta.parquet.tmp"
    if family == "sdxl":
        checkpoint_delta_filename = "checkpoint_delta.npz"
        checkpoint_delta_tmp_filename = "checkpoint_delta.npz.tmp"

    legacy_archive_dir = None
    legacy_archive_prefix = None
    legacy_checkpoint_delta_path = None
    if family == "sd":
        legacy_archive_dir = os.path.join(output_dir, "clip_archive")
        legacy_archive_prefix = "clip_archive_"
        legacy_checkpoint_delta_path = os.path.join(
            output_dir, "clip_checkpoint_delta.parquet"
        )

    return TargetLayout(
        family=family,
        root_dir=root_dir,
        archive_dir=os.path.join(root_dir, "archive"),
        archive_prefix=spec.archive_prefix,
        checkpoint_delta_path=os.path.join(root_dir, checkpoint_delta_filename),
        checkpoint_delta_tmp_path=os.path.join(root_dir, checkpoint_delta_tmp_filename),
        errors_path=os.path.join(root_dir, "errors.parquet"),
        errors_tmp_path=os.path.join(root_dir, "errors.parquet.tmp"),
        manifest_path=os.path.join(root_dir, "manifest.json"),
        best_checkpoint_path=os.path.join(root_dir, f"qwen_{family}_projector_best.pt"),
        gguf_path=os.path.join(resolved_output_dir, f"qwen_{family}_projector.gguf"),
        legacy_archive_dir=legacy_archive_dir,
        legacy_archive_prefix=legacy_archive_prefix,
        legacy_checkpoint_delta_path=legacy_checkpoint_delta_path,
    )


def ensure_target_root(layout):
    os.makedirs(layout.root_dir, exist_ok=True)
    os.makedirs(layout.archive_dir, exist_ok=True)


def build_target_manifest(
    family, *, dtype="float32", shard_size=None, prompt_column=None
):
    spec = get_target_family_spec(family)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "family": spec.family,
        "display_name": spec.display_name,
        "dtype": dtype,
        "shard_size": shard_size,
    }
    if prompt_column is not None:
        payload["prompt_column"] = prompt_column
    payload.update(asdict(spec))
    return payload


def load_target_manifest(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_target_manifest(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
