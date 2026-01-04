URGENT_KEYWORDS = [
    "urgent", "immediately", "asap", "critical",
    "not working", "system down", "blocked", "unable to access"
]

def detect_urgent_intent(text: str):
    """
    Returns True if urgent intent is detected
    """
    text = text.lower()
    return any(keyword in text for keyword in URGENT_KEYWORDS)
