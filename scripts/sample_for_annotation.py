import pandas as pd
import os
from sklearn.model_selection import train_test_split

os.makedirs('../data/annotated', exist_ok=True)
df = pd.read_csv('../data/cleaned/cleaned_dataset.csv')

text_col = 'clean_text'
label_col = None

print("Using text column:", text_col)
print("Label column:", label_col)

if label_col:
    seed, _ = train_test_split(df, train_size=0.1, stratify=df[label_col], random_state=42)
else:
    seed = df.sample(n=min(2000, len(df)), random_state=42)

seed.to_json('../data/annotated/seed_for_labeling.jsonl', orient='records', lines=True)
print("Seed saved to ../data/annotated/seed_for_labeling.jsonl")
