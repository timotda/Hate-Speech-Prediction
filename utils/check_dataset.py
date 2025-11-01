import pandas as pd
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Check label balance in datasets.")
    parser.add_argument("--dataset_path", required=True, help="Dataset directory path")
    args = parser.parse_args()
    return args.dataset_path

def main():
    dataset_dir = Path(parse_args())
    data_files = list(dataset_dir.glob('*.csv'))
    if not data_files:
        print(f"No CSV files found in {dataset_dir}")
        return

    total_pos = 0
    total_neg = 0
    total = 0

    for data_file in data_files:
        print(f"Processing {data_file}...")
        df = pd.read_csv(data_file)
        df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
        labels = df["label"].astype("float32")
        # Only keep 0 and 1 labels
        valid = labels.isin([0, 1])
        labels = labels[valid]
        pos = (labels == 1).sum()
        neg = (labels == 0).sum()
        print(f"  Hate speech: {pos}, Non-hate speech: {neg}, Total: {len(labels)}")
        total_pos += pos
        total_neg += neg
        total += len(labels)

    print("\n=== Overall statistics ===")
    print(f"Total samples: {total}")
    print(f"Total hate speech: {total_pos} ({100*total_pos/total:.2f}%)")
    print(f"Total non-hate speech: {total_neg} ({100*total_neg/total:.2f}%)")

if __name__ == "__main__":
    main()