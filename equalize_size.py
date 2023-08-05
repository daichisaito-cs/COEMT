import csv
import random

def read_data(input_file):
    data = {}
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = int(row['score'])
            if score not in data:
                data[score] = []
            data[score].append(row)
    return data

def equalize_data(data):
    min_length = min(len(v) for v in data.values())
    return {k: random.sample(v, min_length) for k, v in data.items()}

def write_data(output_file, data):
    with open(output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['imgid', 'src', 'mt', 'ref', 'score'])
        writer.writeheader()
        for score in data:
            for row in data[score]:
                writer.writerow(row)

input_file = 'data/shichimi_train_da.csv'  # 入力ファイル
output_file = 'data/shichimi_train_same_size.csv'  # 出力ファイル

data = read_data(input_file)
data = equalize_data(data)
write_data(output_file, data)
