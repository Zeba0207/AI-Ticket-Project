import re
from html import unescape
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------------------
# SAFE NLTK RESOURCE LOADING (DEPLOYMENT)
# --------------------------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1️⃣ Decode HTML & lowercase
    text = unescape(text).lower()

    # 2️⃣ Remove emails, phone numbers, IPs (PII masking)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', text)
    text = re.sub(r'\b\d{10}\b', ' ', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', ' ', text)

    # 3️⃣ Remove punctuation & special characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # 4️⃣ Tokenization (lightweight, deployment-safe)
    tokens = text.split()

    # 5️⃣ Remove stopwords & short tokens
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # 6️⃣ Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # 7️⃣ Remove duplicates (order preserved)
    tokens = list(dict.fromkeys(tokens))

    return " ".join(tokens)
