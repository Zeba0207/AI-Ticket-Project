import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[1]/"data/cleaned/cleaned_dataset.csv")
print(df["category"].value_counts(normalize=True))
