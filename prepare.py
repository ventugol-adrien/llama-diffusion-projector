import pandas as pd
import os


def prepare_longclip_diffusiondb():
    print("Downloading DiffusionDB metadata (no images)...")
    url = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet"
    df = pd.read_parquet(url)

    print("Sanitizing data...")
    df = df[["prompt"]].copy()
    df["prompt"] = df["prompt"].str.lower()
    df = df.drop_duplicates(subset=["prompt"])

    # ==========================================
    # THE LONGCLIP LENGTH FILTER
    # ==========================================
    # We raise the ceiling from 70 words to 170 words.
    # This safely keeps the text under LongCLIP's 248-token limit.
    df["word_count"] = df["prompt"].apply(lambda x: len(str(x).split()))
    df = df[(df["word_count"] >= 3) & (df["word_count"] <= 170)]

    df = df.dropna(subset=["prompt"])

    # Take exactly 1,000,000 random prompts for our training loop
    if len(df) >= 1000000:
        df_final = df.sample(n=1000000, random_state=42).reset_index(drop=True)
    else:
        df_final = df.reset_index(drop=True)

    output_file = "longclip_training_prompts.parquet"
    df_final[["prompt"]].to_parquet(output_file)

    print(f"\nSuccess! Saved {len(df_final)} clean LongCLIP prompts to {output_file}.")


if __name__ == "__main__":
    prepare_longclip_diffusiondb()
