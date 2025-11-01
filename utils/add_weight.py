import csv

input_file = 'data/train_prefixes_fine_tuning.csv'
output_file = 'data/train_prefixes_fine_tuning_with_weight.csv'

# Step 1: Read all rows and group by phrase_id
rows = []
max_word_counts = {}

with open(input_file, 'r', encoding='utf-8', newline='') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        phrase_id = row['phrase_id']
        phrase = row['partial_phrase']
        word_count = len(phrase.split())
        rows.append({**row, 'word_count': word_count})
        if phrase_id not in max_word_counts or word_count > max_word_counts[phrase_id]:
            max_word_counts[phrase_id] = word_count

# Step 2: Compute weight and write to new CSV
fieldnames = list(rows[0].keys())
fieldnames.remove('word_count')
fieldnames.append('weight')

with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        phrase_id = row['phrase_id']
        weight = row['word_count'] / max_word_counts[phrase_id] if max_word_counts[phrase_id] > 0 else 0.0
        out_row = {k: row[k] for k in fieldnames if k != 'weight'}
        out_row['weight'] = weight
        writer.writerow(out_row)

print(f"Done! Output written to {output_file}")