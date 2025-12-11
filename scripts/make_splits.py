import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

data_path = Path(__file__).resolve().parents[1]/"data"/"cleaned"/"cleaned_dataset.csv"
df = pd.read_csv(data_path)

df = df.dropna(subset=["category","text_clean"])

train_df, test_df = train_test_split(df,test_size=0.2,random_state=42,stratify=df["category"])
train_df, val_df = train_test_split(train_df,test_size=0.1,random_state=42,stratify=train_df["category"])

base = Path(__file__).resolve().parents[1]/"data"/"splits"
base.mkdir(parents=True,exist_ok=True)

train_df.to_csv(base/"train.csv",index=False)
val_df.to_csv(base/"val.csv",index=False)
test_df.to_csv(base/"test.csv",index=False)

print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)
