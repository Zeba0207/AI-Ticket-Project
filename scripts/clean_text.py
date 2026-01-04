import re
from html import unescape
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources (first time only)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1️⃣ Decode HTML & lowercase
    text = unescape(text).lower()

    # 2️⃣ Remove emails, phone numbers, IPs
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', text)
    text = re.sub(r'\b\d{10}\b', ' ', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', ' ', text)

    # 3️⃣ Remove punctuation & special characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # 4️⃣ Tokenization
    tokens = text.split()

    # 5️⃣ Remove stopwords & very short tokens
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # 6️⃣ Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # 7️⃣ Remove duplicate tokens (optional but useful)
    tokens = list(dict.fromkeys(tokens))

    return " ".join(tokens)
