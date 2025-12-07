import pandas as pd
from sklearn.model_selection import train_test_split

# load cleaned dataset
df = pd.read_csv("../data/cleaned/cleaned_dataset.csv")

# remove rows with missing category or priority
df = df.dropna(subset=["category", "priority", "clean_text"])

# split (80% train, 20% test)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["category"]   # keep balance by category
)

# save splits
train_df.to_csv("../data/splits/train.csv", index=False)
test_df.to_csv("../data/splits/test.csv", index=False)

print("Train size:", train_df.shape)
print("Test size:", test_df.shape)
print("Saved train.csv and test.csv in data/splits/")
