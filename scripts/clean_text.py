import re
from html import unescape

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = unescape(text)
    text = text.lower()

    # Remove personal info (privacy-safe)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', text)
    text = re.sub(r'\b\d{10}\b', ' ', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', ' ', text)

    # Keep letters + numbers (IMPORTANT for errors, versions)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
