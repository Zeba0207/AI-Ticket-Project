import re

def extract_entities(text: str):
    """
    Lightweight entity extraction using patterns
    (sufficient for academic + real-world baseline)
    """

    usernames = re.findall(r'\buser[_-]?\w+\b', text.lower())
    devices = re.findall(
        r'\b(laptop|desktop|mouse|keyboard|printer|monitor|router)\b',
        text.lower()
    )
    error_codes = re.findall(r'\b(error|err|code)\s?\d+\b', text.lower())

    return {
        "usernames": list(set(usernames)),
        "devices": list(set(devices)),
        "error_codes": list(set(error_codes))
    }
