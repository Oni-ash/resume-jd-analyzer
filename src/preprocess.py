"""
Text preprocessing: clean, tokenize, remove stopwords, lemmatize.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import wordninja

# Download required NLTK data on first run
for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Basic cleanup: remove weird chars, collapse whitespace."""
    # Remove (cid:xxx) artifacts from pdfplumber
    text = re.sub(r"\(cid:\d+\)", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove non-ASCII
    text = text.encode("ascii", "ignore").decode()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_glued_words(text: str, min_len: int = 18) -> str:
    """
    Split long glued-together word runs that PDF extraction sometimes produces.
    E.g. 'cybersecurityenthusiastwithhands' -> 'cyber security enthusiast with hands'
    Short PDF glue (like 'craftingsystem') is left alone and handled by
    substring fallback in skill_extract.
    """
    def _split(match: re.Match) -> str:
        token = match.group(0)
        if len(token) >= min_len:
            return " ".join(wordninja.split(token))
        return token

    return re.sub(r"[a-zA-Z]+", _split, text)


def tokenize(text: str) -> list[str]:
    """Split into lowercase word tokens (letters only)."""
    return re.findall(r"[a-zA-Z]+", text.lower())


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _stop_words and len(t) > 1]


def lemmatize(tokens: list[str]) -> list[str]:
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str, as_string: bool = False):
    """
    Full pipeline: clean -> fix glued words -> tokenize -> stopwords -> lemmatize.
    Returns list of tokens by default, or a joined string if as_string=True.
    """
    text = clean_text(text)
    text = fix_glued_words(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    if as_string:
        return " ".join(tokens)
    return tokens


if __name__ == "__main__":
    import sys
    from src.extract import extract_text

    if len(sys.argv) < 2:
        print("Usage: python -m src.preprocess <file_path>")
        sys.exit(1)

    raw = extract_text(sys.argv[1])
    tokens = preprocess(raw)
    print(f"--- {len(tokens)} tokens after preprocessing ---")
    print(" ".join(tokens[:80]), "...")