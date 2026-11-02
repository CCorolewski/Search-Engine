#Author: Caleb Corolewski
#Date: Fall 2025
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#------------------------------ PREPROCESSING -------------------------------
TERM_META = {
    "cacm", "html", "pre",
    "january","february","march","april","may","june",
    "july","august","september","october","november","december"
}

# precompute stemmed meta terms to filter after stemming
_meta_stemmer = PorterStemmer()
STEMMED_META = {_meta_stemmer.stem(t) for t in TERM_META}


def standard_preprocess(text: str) -> list[str]:
    """
    Standard preprocessing for the vector model:
    - lowercase
    - strip HTML to plain text
    - tokenize
    - remove stopwords
    - remove TERM_META terms
    - keep tokens with at least one alphabetic character
    - stem with PorterStemmer
    """
    # Strip HTML if present (safe for plain text too)
    try:
        clean = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except Exception:
        # fallback: just use the original text, no regex
        clean = text

    # Lowercase + tokenize
    tokens = word_tokenize(clean.lower())

    # Remove stopwords
    stops = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    terms: list[str] = []
    for t in tokens:
        # must contain at least one alphabetic char
        if not any(c.isalpha() for c in t):
            continue
        # drop stopwords
        if t in stops:
            continue

        stem = stemmer.stem(t)

        # drop TERM_META (months, 'cacm', 'html', etc.)
        if stem in STEMMED_META:
            continue

        # drop extremely short stems (2 char)
        if len(stem) <= 2:
            continue

        terms.append(stem)

    return terms
    
#-------------------------- PREPROCESSOR FOR QLM ------------------------
def ql_preprocess(text: str) -> list[str]:
    """
    Preprocessing for Query Likelihood model.
    - Strip HTML using BeautifulSoup
    - Lowercase
    - Tokenize
    - KEEP stopwords
    - Require at least one alphabetic character
    - Stem with PorterStemmer
    """

    # Strip HTML
    try:
        clean = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except Exception:
        # If BeautifulSoup fails, just fall back to the raw text
        clean = text

    tokens = word_tokenize(clean.lower())
    stemmer = PorterStemmer()
    terms: list[str] = []

    for t in tokens:
        # keep only tokens containing at least one alphabetic character
        has_alpha = False
        for c in t:
            if c.isalpha():
                has_alpha = True
                break
        if not has_alpha:
            continue

        stem = stemmer.stem(t)

        # drop short stems (noise)
        if len(stem) < 2:
            continue

        terms.append(stem)

    return terms