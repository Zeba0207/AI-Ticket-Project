import re
import spacy

nlp = spacy.load("en_core_web_sm")

# Regex patterns for IT tickets
ERROR_PATTERN = r"\b(err(or)?|code)\s?\d+\b|\b0x[a-fA-F0-9]+\b"
DEVICE_PATTERN = r"\b(laptop|desktop|printer|server|router|pc)\b"

def extract_entities(text: str):
    """
    Extracts entities like usernames, devices, and error codes
    """
    doc = nlp(text)

    entities = {
        "usernames": [],
        "devices": [],
        "error_codes": []
    }

    # Named entities from spaCy
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["usernames"].append(ent.text)

    # Regex-based entities
    entities["devices"] = re.findall(DEVICE_PATTERN, text.lower())
    entities["error_codes"] = re.findall(ERROR_PATTERN, text.lower())

    return entities
