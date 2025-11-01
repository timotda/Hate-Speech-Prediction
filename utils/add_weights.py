import pandas as pd
import argparse

def calculate_and_add_weights(input_filepath, output_filepath):
    """
    Reads a CSV file, calculates the weight for each text prefix relative to its full phrase,
    and adds this weight in a new column.

    The weight is (number of words in prefix) / (number of words in the full phrase for that index).
    The full phrase for an index is the text entry with the maximum number of words for that index.

    Args:
        input_filepath (str): Path to the input CSV file.
                               Expected columns: 'index', 'text', and others.
        output_filepath (str): Path to save the output CSV file with the new 'weight' column.
    """
    try:
        # Read the input CSV
        df = pd.read_csv(input_filepath)

        # Ensure 'text' column is treated as string, handling potential NaNs
        df['text'] = df['text'].astype(str)

        # Calculate the number of words in each 'text' entry (prefix)
        df['prefix_word_count'] = df['text'].apply(lambda x: len(x.split()))

        # Determine the maximum number of words for each 'index' (length of the full phrase)
        # .transform('max') broadcasts this maximum value to all rows belonging to the same 'index'
        df['full_phrase_word_count'] = df.groupby('index')['prefix_word_count'].transform('max')

        # Calculate the weight
        # Avoid division by zero if a full phrase somehow has 0 words (e.g., all texts for an index are empty)
        # In such cases, weight will be NaN, which we then convert to 0.
        df['weight'] = df['prefix_word_count'] / df['full_phrase_word_count']
        
        # Handle potential NaN (from 0/0) or inf (if full_phrase_word_count was 0 and prefix_word_count > 0, though less likely here)
        # by replacing them with 0.
        df['weight'].replace([float('inf'), -float('inf')], 0, inplace=True)
        df['weight'].fillna(0, inplace=True)

        # Select columns for output: all original columns + new 'weight' column
        # Drop the intermediate calculation columns
        df_output = df.drop(columns=['prefix_word_count', 'full_phrase_word_count'])

        # Save the updated DataFrame to a new CSV file
        df_output.to_csv(output_filepath, index=False)
        print(f"Successfully processed '{input_filepath}' and saved results to '{output_filepath}'.")
        print(f"The new 'weight' column has been added.")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'")
    except KeyError as e:
        print(f"Error: Missing expected column in CSV: {e}. Ensure 'index' and 'text' columns exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculates prefix weights from a CSV file and adds them as a new 'weight' column."
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file (e.g., data/test_prefixes.csv)."
    )
    parser.add_argument(
        "output_csv",
        help="Path to save the output CSV file with the new 'weight' column."
    )

    args = parser.parse_args()

    calculate_and_add_weights(args.input_csv, args.output_csv)

# --- How to use it ---
# 1. Save the script: Save the code above as a Python file (e.g., `calculate_weights.py`).
# 2. Make sure you have pandas installed: `pip install pandas`
# 3. Run from the terminal:
#
#    python add_weights.py path/to/your/input.csv path/to/your/output.csv
#
# Example using the provided file:
#    python add_weights.py /home/federico/EPFL/DeepLearningProject/EPFL-EE559-HateSpeechDetection/data/test_prefixes.csv /home/federico/EPFL/DeepLearningProject/EPFL-EE559-HateSpeechDetection/data/test_prefixes_with_weights.csv
#
# This will create 'test_prefixes_with_weights.csv' with the original columns and the new 'weight' column.
# The 'label' column from your input 'test_prefixes.csv' will be preserved, and the newly calculated
# 'weight' will be in its own column.