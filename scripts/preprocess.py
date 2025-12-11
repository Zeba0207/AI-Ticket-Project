import re
import pandas as pd
from html import unescape
import spacy
from pathlib import Path
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
tqdm.pandas()

# Map inconsistent category names
CATEGORY_MAP = {
    "account/access issue": "account_access_issue",
    "Hardware Issue": "hardware_issue",
    "network problem": "network_problem",
    "security": "security",
    "Service Request": "service_request",
    "software bug": "software_bug",
    "other": "other"
}

def normalize_category(cat):
    if not isinstance(cat, str):
        return None
    cat = cat.strip()
    return CATEGORY_MAP.get(cat, None)  # return mapped or None

def mask_pii(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '<EMAIL>', text)
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{2,4}[-.\s]?){2,4}\d{2,4}\b','<PHONE>',text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>', text)
    return text

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = unescape(text).lower()
    text = text.replace("\r"," ").replace("\t"," ")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = mask_pii(text)
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if not t.is_stop]
    return re.sub(r'\s+',' '," ".join(tokens)).strip()

data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "final_dataset_utf8.csv"
df = pd.read_csv(data_path)

df["category"] = df["category"].apply(normalize_category)
df["text_clean"] = df["text"].progress_apply(clean_text)

df = df.dropna(subset=["category", "text_clean"])
df = df.drop_duplicates()

output_path = Path(__file__).resolve().parents[1]/"data"/"cleaned"/"cleaned_dataset.csv"
output_path.parent.mkdir(parents=True,exist_ok=True)
df.to_csv(output_path,index=False)

# Save distribution
dist_path = Path(__file__).resolve().parents[1]/"data"/"cleaned"/"category_distribution.csv"
df["category"].value_counts().to_csv(dist_path)

print("Saved cleaned dataset to:", output_path)
print("Category distribution saved to:", dist_path)
