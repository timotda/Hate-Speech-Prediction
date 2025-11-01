import csv

input_file = 'data\\olid-training-v1.0.tsv'   # Replace with your input file name
output_file = 'data\\data_olid.csv' # Replace with your desired output file name

with open(input_file, 'r', encoding='utf-8', newline='') as infile, \
     open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.DictReader(infile, delimiter='\t')
    fieldnames = ['text', 'label']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in reader:
        label = '1' if row.get('subtask_b') == 'TIN' else '0'
        writer.writerow({'text': row['tweet'], 'label': label})

print(f"Done! Output written to {output_file}")