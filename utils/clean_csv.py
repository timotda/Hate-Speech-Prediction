import csv
import argparse
import os

def clean_csv_column(input_filepath, output_filepath, column_to_clean, term_to_remove="@USER"):
    """
    Reads a CSV file, removes all occurrences of a specific term (e.g., "@USER")
    from a specified column, and writes the cleaned data to a new CSV file.

    Args:
        input_filepath (str): Path to the input CSV file.
        output_filepath (str): Path to save the cleaned CSV file.
        column_to_clean (str): The name of the column from which to remove the term.
        term_to_remove (str, optional): The exact string to remove. Defaults to "@USER".
    """
    try:
        cleaned_rows = 0
        rows_processed = 0

        with open(input_filepath, mode='r', newline='', encoding='utf-8') as infile, \
             open(output_filepath, mode='w', newline='', encoding='utf-8') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # Read the header
            header = next(reader, None)
            if not header:
                print("Error: Input CSV file is empty or has no header.")
                return

            writer.writerow(header) # Write header to output

            # Find the index of the column to clean
            try:
                column_index = header.index(column_to_clean)
            except ValueError:
                print(f"Error: Column '{column_to_clean}' not found in the CSV header.")
                print(f"Available columns are: {', '.join(header)}")
                # Copy the rest of the file as is if column not found, or just exit
                # For now, let's just copy the rest
                for row in reader:
                    writer.writerow(row)
                print(f"Warning: Column '{column_to_clean}' not found. Copied file without changes to '{output_filepath}'.")
                return

            print(f"Processing column '{column_to_clean}' (index {column_index})...")

            # Process each row
            for row in reader:
                rows_processed += 1
                if len(row) > column_index:
                    original_text = row[column_index]
                    # Ensure the value is a string before calling replace
                    if isinstance(original_text, str):
                        cleaned_text = original_text.replace(term_to_remove, "")
                        cleaned_text = original_text.replace(";", "")
                        cleaned_text.strip()  # Remove leading/trailing whitespace
                        if original_text != cleaned_text:
                            cleaned_rows +=1
                        row[column_index] = cleaned_text
                    # else:
                        # print(f"Warning: Row {rows_processed+1}, Column '{column_to_clean}' is not a string: {original_text}")
                # else:
                    # print(f"Warning: Row {rows_processed+1} is shorter than expected, skipping column cleaning for this row.")

                writer.writerow(row)

        print(f"\nSuccessfully processed {rows_processed} data rows.")
        print(f"Removed '{term_to_remove}' from {cleaned_rows} entries in column '{column_to_clean}'.")
        print(f"Cleaned data saved to: {output_filepath}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Cleans a CSV dataset by removing all occurrences of a specified term (default: '@USER') from a given text column.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "output_csv",
        help="Path to save the cleaned CSV file."
    )
    parser.add_argument(
        "column_name",
        help="Name of the text column to clean (case-sensitive)."
    )
    parser.add_argument(
        "--term",
        default="@USER",
        help="The specific term to remove (default: '@USER')."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists."
    )

    args = parser.parse_args()

    if not args.force and os.path.exists(args.output_csv):
        response = input(f"Output file '{args.output_csv}' already exists. Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Operation cancelled by user.")
            exit()

    clean_csv_column(args.input_csv, args.output_csv, args.column_name, args.term)
