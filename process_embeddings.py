import pandas as pd
import sys


def calculate_kb_payload_size():
    dataset_path = "longclip_training_prompts.parquet"
    print(f"Loading dataset: {dataset_path}...")
    df = pd.read_parquet(dataset_path)

    # We need the raw strings to measure their byte size
    prompts = df["prompt"].tolist()

    print("Calculating UTF-8 Byte sizes for all prompts...")

    # Calculate exact byte size for every string using UTF-8 encoding
    byte_sizes = [len(text.encode("utf-8")) for text in prompts]

    max_bytes = max(byte_sizes)
    avg_bytes = sum(byte_sizes) / len(byte_sizes)
    total_bytes = sum(byte_sizes)

    # Convert to Kilobytes (KB)
    max_kb = max_bytes / 1024
    avg_kb = avg_bytes / 1024
    total_mb = total_bytes / (1024 * 1024)

    print("\n" + "=" * 40)
    print("💾 DATASET PAYLOAD METRICS (UTF-8)")
    print("=" * 40)
    print(f"Total Prompts:      {len(prompts):,}")
    print(f"Total Dataset Size: {total_mb:.2f} MB (Text Only)")
    print("-" * 40)
    print(f"Average Prompt:     {avg_bytes:.0f} Bytes ({avg_kb:.3f} KB)")
    print(f"Absolute Longest:   {max_bytes} Bytes ({max_kb:.3f} KB)")
    print("=" * 40)

    # Find the actual string that is the largest to inspect it
    max_index = byte_sizes.index(max_bytes)
    print(f"\n📝 The heaviest prompt ({max_kb:.3f} KB) looks like this:")
    print(f'"{prompts[max_index]}"')


if __name__ == "__main__":
    calculate_kb_payload_size()
