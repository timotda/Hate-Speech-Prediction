import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

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

    # Initialize counters for overall statistics
    overall_rows_read = 0
    overall_rows_after_initial_dropna = 0
    overall_invalid_label_type = 0
    overall_invalid_label_range = 0
    overall_invalid_weight_type = 0
    overall_invalid_weight_range = 0
    overall_empty_text = 0
    overall_fully_valid_rows = 0

    label_counts = {}

    required_columns = ["text", "label", "weight"]

    for data_file in data_files:
        print(f"Processing {data_file}...")
        try:
            df = pd.read_csv(data_file)
        except Exception as e:
            print(f"  Error reading CSV {data_file}: {e}")
            continue
        
        rows_in_file_initial = len(df)
        overall_rows_read += rows_in_file_initial
        print(f"  Rows read: {rows_in_file_initial}")

        # Check for missing required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"  Missing required columns: {', '.join(missing_cols)}. Skipping file.")
            continue

        # Initial dropna for required columns
        df_cleaned = df.dropna(subset=required_columns).copy()
        rows_after_initial_dropna = len(df_cleaned)
        overall_rows_after_initial_dropna += rows_after_initial_dropna
        print(f"  Rows after dropping NaNs in {required_columns}: {rows_after_initial_dropna} (dropped {rows_in_file_initial - rows_after_initial_dropna})")

        if df_cleaned.empty:
            print("  No rows remaining after initial NaN drop.")
            print("  ---")
            continue
    
        # Dropping lines with label == 0.0
        #df_cleaned = df_cleaned[df_cleaned['label'] != 0.0]

        # --- Text validation ---
        df_cleaned['text'] = df_cleaned['text'].astype(str)
        empty_text_mask = df_cleaned['text'].str.strip() == ""
        num_empty_text_in_file = empty_text_mask.sum()
        overall_empty_text += num_empty_text_in_file
        if num_empty_text_in_file > 0:
            print(f"  Found {num_empty_text_in_file} rows with empty 'text'.")
        
        # --- Label validation ---
        original_labels_before_coerce = df_cleaned['label'].copy()
        df_cleaned['label'] = pd.to_numeric(df_cleaned['label'], errors='coerce')
        
        non_numeric_label_mask = df_cleaned['label'].isna() & original_labels_before_coerce.notna()
        num_invalid_label_type_in_file = non_numeric_label_mask.sum()
        overall_invalid_label_type += num_invalid_label_type_in_file
        if num_invalid_label_type_in_file > 0:
            print(f"  Found {num_invalid_label_type_in_file} non-numeric 'label' values (set to NaN).")

        numeric_labels = df_cleaned.loc[df_cleaned['label'].notna(), 'label']
        invalid_label_range_mask = ~numeric_labels.between(0, 1, inclusive='both')
        num_invalid_label_range_in_file = invalid_label_range_mask.sum()
        overall_invalid_label_range += num_invalid_label_range_in_file
        if num_invalid_label_range_in_file > 0:
            print(f"  Found {num_invalid_label_range_in_file} numeric 'label' values outside [0, 1] range.")
        
        # --- Count label occurrences ---
        for label in numeric_labels.unique():
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += (numeric_labels == label).sum()

        # --- Weight validation ---
        original_weights_before_coerce = df_cleaned['weight'].copy()
        df_cleaned['weight'] = pd.to_numeric(df_cleaned['weight'], errors='coerce')

        non_numeric_weight_mask = df_cleaned['weight'].isna() & original_weights_before_coerce.notna()
        num_invalid_weight_type_in_file = non_numeric_weight_mask.sum()
        overall_invalid_weight_type += num_invalid_weight_type_in_file
        if num_invalid_weight_type_in_file > 0:
            print(f"  Found {num_invalid_weight_type_in_file} non-numeric 'weight' values (set to NaN).")

        numeric_weights = df_cleaned.loc[df_cleaned['weight'].notna(), 'weight']
        invalid_weight_range_mask = ~numeric_weights.between(0, 1, inclusive='both')
        num_invalid_weight_range_in_file = invalid_weight_range_mask.sum()
        overall_invalid_weight_range += num_invalid_weight_range_in_file
        if num_invalid_weight_range_in_file > 0:
            print(f"  Found {num_invalid_weight_range_in_file} numeric 'weight' values outside [0, 1] range.")
        
        # --- Determine fully valid rows for this file ---
        valid_rows_mask = ~empty_text_mask & \
                          df_cleaned['label'].notna() & df_cleaned['label'].between(0, 1, inclusive='both') & \
                          df_cleaned['weight'].notna() & df_cleaned['weight'].between(0, 1, inclusive='both')
        
        num_fully_valid_in_file = valid_rows_mask.sum()
        overall_fully_valid_rows += num_fully_valid_in_file
        print(f"  Fully valid rows in this file: {num_fully_valid_in_file}")
        print("  ---")

    print("\n=== Overall Validation Statistics ===")
    print(f"Total CSV files found: {len(data_files)}")
    print(f"Total rows read across all files: {overall_rows_read}")
    print(f"Total rows after initial NaN drop in {required_columns}: {overall_rows_after_initial_dropna}")
    print(f"Total rows with non-numeric 'label' values: {overall_invalid_label_type}")
    print(f"Total rows with numeric 'label' values outside [0, 1]: {overall_invalid_label_range}")
    print(f"Total rows with non-numeric 'weight' values: {overall_invalid_weight_type}")
    print(f"Total rows with numeric 'weight' values outside [0, 1]: {overall_invalid_weight_range}")
    print(f"Total rows with empty 'text' (after initial NaN drop): {overall_empty_text}")
    print(f"Total fully valid rows (meeting all criteria): {overall_fully_valid_rows}")
    print(f"  Label counts: {label_counts}")
    plt.bar(label_counts.keys(), label_counts.values(), width=0.09)
    plt.xlabel('Label', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.title('Label Distribution', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(dataset_dir / "label_distribution.png")
    plt.show()

    if overall_rows_read > 0:
        percentage_valid = (overall_fully_valid_rows / overall_rows_read) * 100
        print(f"Percentage of fully valid rows from total read: {percentage_valid:.2f}%")
    if overall_rows_after_initial_dropna > 0 and overall_rows_after_initial_dropna > 0 : # check for overall_rows_after_initial_dropna > 0 to avoid division by zero
        percentage_valid_post_dropna = (overall_fully_valid_rows / overall_rows_after_initial_dropna) * 100
        print(f"Percentage of fully valid rows from rows that passed initial NaN drop: {percentage_valid_post_dropna:.2f}%")


if __name__ == "__main__":
    main()