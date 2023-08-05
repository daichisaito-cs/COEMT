import pandas as pd
from janome.tokenizer import Tokenizer
import nltk

nltk.download('punkt')

# CSVファイルを読み込む
df = pd.read_csv('data/shichimi_train_da.csv')
df_val = pd.read_csv("data/shichimi_val_da.csv")

# 全文を取得
sentences = (df["ref"]).tolist() + df["mt"].tolist() + df["src"].tolist() + (df_val["ref"]).tolist() + df_val["mt"].tolist() + df_val["src"].tolist()

t = Tokenizer()

# 全文をトークン化（単語に分割）
tokens = [list(t.tokenize(sentence, wakati=True)) for sentence in sentences]

# 全単語数
total_words = sum(len(token) for token in tokens)

# 語彙サイズ（ユニーク語数）
vocab_size = len(set(word for sublist in tokens for word in sublist))

# 平均文長
avg_sentence_length = total_words / len(sentences)

print(f'Vocabulary Size: {vocab_size}')
print(f'Total Words: {total_words}')
print(f'Average Sentence Length: {avg_sentence_length}')
