import re
import pandas as pd
from html import unescape
import spacy
from tqdm import tqdm

# load spaCy model once
nlp = spacy.load("en_core_web_sm")

tqdm.pandas()  # progress bar for pandas apply


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
    """Complete cleaning pipeline with lemmatization."""
    if not isinstance(text, str):
        return ""

    text = unescape(text)
    text = text.replace('\r', ' ').replace('\t', ' ')
    text = mask_pii(text)

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    ]

    cleaned = " ".join(tokens)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned
