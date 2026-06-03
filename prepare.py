import os

import pandas as pd

DEFAULT_PROMPT_COLUMN = "prompt"
RAW_PROMPT_COLUMN = "prompt_raw"
NORMALIZED_PROMPT_COLUMN = "prompt_normalized"


def _resolve_output_prompt_variant() -> tuple[str, str]:
    variant = os.getenv("PREPARE_OUTPUT_PROMPT_VARIANT", "normalized").strip().lower()
    prompt_columns = {
        "normalized": NORMALIZED_PROMPT_COLUMN,
        "raw": RAW_PROMPT_COLUMN,
    }
    if variant not in prompt_columns:
        supported = ", ".join(sorted(prompt_columns))
        raise ValueError(
            "PREPARE_OUTPUT_PROMPT_VARIANT must be one of: "
            f"{supported}. Got '{variant}'."
        )
    return variant, prompt_columns[variant]


def _empty_prompt_count(row_count: int) -> int:
    ratio = float(os.getenv("PREPARE_EMPTY_PROMPT_RATIO", "0.0"))
    if ratio < 0:
        raise ValueError("PREPARE_EMPTY_PROMPT_RATIO must be >= 0.")
    if ratio == 0 or row_count == 0:
        return 0
    return max(1, int(round(row_count * ratio)))


def prepare_longclip_diffusiondb():
    output_file = os.getenv("PREPARE_OUTPUT_FILE", "longclip_training_prompts.parquet")
    sample_size = max(1, int(os.getenv("PREPARE_SAMPLE_SIZE", "1000000")))
    prompt_variant, output_prompt_column = _resolve_output_prompt_variant()

    print("Downloading DiffusionDB metadata (no images)...")
    url = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet"
    df = pd.read_parquet(url)

    print("Sanitizing data...")
    df = df[["prompt"]].copy()
    df = df.dropna(subset=["prompt"])
    df[RAW_PROMPT_COLUMN] = df["prompt"].astype(str).str.strip()
    df = df[df[RAW_PROMPT_COLUMN] != ""].copy()
    df[NORMALIZED_PROMPT_COLUMN] = df[RAW_PROMPT_COLUMN].str.lower()
    df = df.drop_duplicates(subset=[NORMALIZED_PROMPT_COLUMN])

    # ==========================================
    # THE LONGCLIP LENGTH FILTER
    # ==========================================
    # We raise the ceiling from 70 words to 170 words.
    # This safely keeps the text under LongCLIP's 248-token limit.
    df["word_count"] = df[RAW_PROMPT_COLUMN].apply(lambda x: len(x.split()))
    df = df[(df["word_count"] >= 3) & (df["word_count"] <= 170)]

    # Take exactly 1,000,000 random prompts for our training loop
    if len(df) >= sample_size:
        df_final = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        df_final = df.reset_index(drop=True)

    df_final[DEFAULT_PROMPT_COLUMN] = df_final[output_prompt_column]
    df_final["is_empty_prompt"] = False

    empty_prompt_count = _empty_prompt_count(len(df_final))
    if empty_prompt_count > 0:
        empty_rows = pd.DataFrame(
            {
                DEFAULT_PROMPT_COLUMN: [""] * empty_prompt_count,
                RAW_PROMPT_COLUMN: [""] * empty_prompt_count,
                NORMALIZED_PROMPT_COLUMN: [""] * empty_prompt_count,
                "is_empty_prompt": [True] * empty_prompt_count,
            }
        )
        df_final = pd.concat(
            [
                df_final[
                    [
                        DEFAULT_PROMPT_COLUMN,
                        RAW_PROMPT_COLUMN,
                        NORMALIZED_PROMPT_COLUMN,
                        "is_empty_prompt",
                    ]
                ],
                empty_rows,
            ],
            ignore_index=True,
        )
    else:
        df_final = df_final[
            [
                DEFAULT_PROMPT_COLUMN,
                RAW_PROMPT_COLUMN,
                NORMALIZED_PROMPT_COLUMN,
                "is_empty_prompt",
            ]
        ]

    df_final.to_parquet(output_file)

    print(
        f"\nSuccess! Saved {len(df_final)} prompts to {output_file} "
        f"(prompt variant: {prompt_variant})."
    )


if __name__ == "__main__":
    prepare_longclip_diffusiondb()
