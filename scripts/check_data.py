import pandas as pd

df = pd.read_csv("../data/annotated/milestone1_labeled.csv")

print("\n--- Columns in the dataset ---")
print(df.columns)

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Shape of dataset (rows, columns) ---")
print(df.shape)
