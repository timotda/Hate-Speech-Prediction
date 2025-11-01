import csv
import argparse

def create_prefix_dataset(input_filepath, output_filepath, text_column_name, label_column_name):
    """
    Reads an input CSV file, generates entries for each prefix of phrases
    in a specified text column, and writes them to a new CSV file.

    For each original entry, it creates multiple new entries: one for each
    possible prefix of the text. Each new entry includes the original label
    and a calculated weight (length of prefix / length of full phrase in words).
    """
    try:
        with open(input_filepath, mode='r', newline='', encoding='utf-8') as infile, \
             open(output_filepath, mode='w', newline='', encoding='utf-8') as outfile:

            reader = csv.DictReader(infile)
            
            # Validate that specified text and label columns exist in the input CSV
            if text_column_name not in reader.fieldnames:
                raise ValueError(f"Text column '{text_column_name}' not found in CSV header. Available columns: {reader.fieldnames}")
            if label_column_name not in reader.fieldnames:
                raise ValueError(f"Label column '{label_column_name}' not found in CSV header. Available columns: {reader.fieldnames}")

            # Define fieldnames for the output CSV. It will include all original columns plus 'weight'.
            # If 'weight' already exists, it will be used (and potentially overwritten by our calculation).
            output_fieldnames = list(reader.fieldnames) # Make a mutable copy
            if 'weight' not in output_fieldnames:
                output_fieldnames.append('weight')

            writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
            writer.writeheader()

            processed_rows_count = 0
            generated_rows_count = 0

            for original_row in reader:
                processed_rows_count += 1
                phrase = original_row.get(text_column_name, "")
                # The label will be carried over from the original_row when we copy it.

                if phrase is None: # Handle cases where the column might exist but contain None
                    phrase = ""

                words = phrase.split()  # Splits by any whitespace and removes empty strings
                num_total_words = len(words)

                if num_total_words == 0:
                    # If the phrase is empty or contains only whitespace, no prefixes can be generated.
                    # According to the logic "creates an entry for each length of the prefixes",
                    # such rows from the input will not result in any output rows.
                    continue

                for i in range(1, num_total_words + 1):
                    prefix_words = words[:i]
                    current_prefix = " ".join(prefix_words)
                    
                    # Weight is length of prefix / length of full phrase
                    weight = i / num_total_words

                    # Create a new row based on the original, then modify/add fields
                    new_row = original_row.copy()
                    new_row[text_column_name] = current_prefix
                    # The label from original_row[label_column_name] is already in new_row
                    new_row['weight'] = weight
                    
                    writer.writerow(new_row)
                    generated_rows_count += 1
            
        print(f"Successfully processed {processed_rows_count} rows from '{input_filepath}'.")
        print(f"Generated {generated_rows_count} rows in '{output_filepath}'.")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Generates a new CSV file with entries for each prefix of a text column. "
                    "Each new entry includes the original label and a calculated weight."
    )
    parser.add_argument("--input_csv", help="Path to the input CSV file.")
    parser.add_argument("--output_csv", help="Path to save the generated CSV file.")
    parser.add_argument(
        "--text_column", 
        default="text", 
        help="Name of the column containing the text phrases (default: 'text')."
    )
    parser.add_argument(
        "--label_column", 
        default="label", 
        help="Name of the column containing the labels (default: 'label')."
    )

    args = parser.parse_args()

    create_prefix_dataset(args.input_csv, args.output_csv, args.text_column, args.label_column)

if __name__ == "__main__":
    main()