import re
import spacy
from textblob import TextBlob

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            _nlp = spacy.blank("en")
    return _nlp

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    return text

def extract_features(text):
    cleaned = clean_text(text)
    doc = get_nlp()(cleaned)

    features = {
        "entities": [ent.label_ for ent in doc.ents],
        "keywords": [token.lemma_ for token in doc if not token.is_stop],
        "sentiment": TextBlob(str(text)).sentiment.polarity,
        "is_refund": 1 if "refund" in cleaned else 0
    }
    return features