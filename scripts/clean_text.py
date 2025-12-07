import re
import pandas as pd
from html import unescape

# ------------ TEXT CLEANING FUNCTIONS ----------------

def mask_pii(text):
    """Mask personal data such as email, phone numbers, and IP addresses."""
    if not isinstance(text, str):
        return ""

    # Mask email
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '<EMAIL>', text)

    # Mask phone numbers
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{2,4}[-.\s]?){2,4}\d{2,4}\b', '<PHONE>', text)

    # Mask IP address
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>', text)

    return text

def clean_text(text):
    """Complete cleaning pipeline."""
    if not isinstance(text, str):
        return ""

    text = unescape(text)  # remove HTML encoding
    text = text.replace('\r', ' ').replace('\t', ' ')
    text = mask_pii(text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip().lower()


# ------------ LOAD DATA & CLEAN ----------------

# UPDATED FILE NAME
input_file = "../data/raw/updated_dataset.csv"   # <--- changed to your new dataset
output_file = "../data/cleaned/cleaned_dataset.csv"

print("Loading dataset...")
df = pd.read_csv(input_file)

# Detect the text column automatically
TEXT_COLUMNS = ["text", "message", "ticket", "description", "query"]

text_col = None
for col in df.columns:
    if col.lower() in TEXT_COLUMNS:
        text_col = col
        break

if text_col is None:
    raise Exception(
        "❌ No text column found. Please rename your main text column to one of: "
        "text / message / ticket / description / query"
    )

print(f"Cleaning text column: {text_col}")

df["clean_text"] = df[text_col].apply(clean_text)

print("Saving cleaned dataset...")
df.to_csv(output_file, index=False)

print("✅ Cleaning complete! Cleaned file saved at:")
print(output_file)
