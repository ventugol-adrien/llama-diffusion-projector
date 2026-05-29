import os

import torch

from target_families import get_target_family, get_target_layout
from train import load_projector_from_gguf

DEFAULT_OUTPUT_DIR = "embedded_chunks"


def run_projector_demo(gguf_path=None, batch_size=2, device=None):
    target_family = get_target_family(
        os.getenv("PROJECTOR_TARGET_FAMILY", os.getenv("TARGET_FAMILY", "sd"))
    )
    if gguf_path is None:
        gguf_path = os.getenv(
            "PROJECTOR_GGUF",
            get_target_layout(DEFAULT_OUTPUT_DIR, target_family).gguf_path,
        )
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(
            f"Projector GGUF not found at {gguf_path}. Run train.py for target_family={target_family} first."
        )

    model, metadata = load_projector_from_gguf(gguf_path, device=device)
    model_device = next(model.parameters()).device
    sample = torch.randn(batch_size, metadata["qwen_dim"], device=model_device)

    with torch.inference_mode():
        outputs = model(sample)

    print(f"Loaded projector: {gguf_path}")
    print(f"Target family: {metadata['target_family']}")
    print(f"Input shape: {tuple(sample.shape)}")
    if metadata["target_family"] == "sdxl":
        prompt_embeds, pooled_prompt_embeds = outputs
        print(f"Prompt output shape: {tuple(prompt_embeds.shape)}")
        print(f"Pooled output shape: {tuple(pooled_prompt_embeds.shape)}")
    else:
        print(f"Output shape: {tuple(outputs.shape)}")
    return outputs, metadata


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        run_projector_demo()
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1)
