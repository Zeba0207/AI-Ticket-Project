import pandas as pd

df = pd.read_csv("data/cleaned/cleaned_dataset.csv")

dist = df["category"].value_counts(normalize=False).reset_index()
dist.columns = ["category", "count"]

# Save file
dist.to_csv("data/cleaned/category_distribution.csv", index=False)

print("Saved category_distribution.csv")
