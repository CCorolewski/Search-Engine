from similarity import Similarity, BM25Similarity, BooleanSimilarity
from typing import List, Tuple
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
#---------------------------- LUCENE STYLE SEARCH -------------------------
class Searcher:
    """
    High-level search, similar in spirit to Lucene's IndexSearcher.
    """

    def __init__(
        self,
        db_path: str = "index.db",
        similarity: Similarity | None = None,
    ):
        self.db_path = db_path
        # default to BM25 if none provided
        self.similarity = similarity if similarity is not None else BM25Similarity()

    def set_similarity(self, similarity: Similarity) -> None:
        self.similarity = similarity

    def search(self, query: str, k: int = 10) -> list[tuple[float, str]]:
        """
        Return top-k (score, filename) using the current Similarity.
        """
        return self.similarity.rank(query=query, db_path=self.db_path, k=k)

    def search_filenames(self, query: str, k: int = 10) -> list[str]:
        """
        Convenience method: returns filenames only.
        """
        ranked = self.search(query, k=k)
        return [fname for score, fname in ranked]

#-------------------------------- SCORE SENTENCE --------------------------
def score_sentence(query_terms: List[Tuple[str, str]], sentence: str) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Score a sentence given (word, stem) tuples.

    Returns:
        (score, spans) where spans is a list of (start, end) character indices (end-exclusive).
    """
    if not sentence or not query_terms:
        return (0, [])

    # Tokenize sentence and reconstruct character spans for each token
    tokens = word_tokenize(sentence)
    lowered_sentence = sentence.lower()
    token_spans: List[Tuple[int, int]] = []
    pos = 0
    for tok in tokens:
        tok_lower = tok.lower()
        start = lowered_sentence.find(tok_lower, pos)
        if start == -1:
            # fallback: search from beginning
            start = lowered_sentence.find(tok_lower)
            if start == -1:
                continue
        end = start + len(tok)
        token_spans.append((start, end))
        pos = end

    # Helper: check overlap between spans
    def overlaps_any(span: Tuple[int, int], spans: List[Tuple[int, int]]) -> bool:
        s, e = span
        for a, b in spans:
            if not (e <= a or b <= s):
                return True
        return False

    word_matches: List[Tuple[int, int]] = []
    stem_matches: List[Tuple[int, int]] = []

    # Exact word matches first (take precedence)
    for (w, _st) in query_terms:
        if not w:
            continue
        w_lower = w.lower()
        for (tok, (s, e)) in zip(tokens, token_spans):
            if tok.lower() == w_lower:
                word_matches.append((s, e))

    # Normalize word matches (sort + dedupe)
    word_matches = sorted(set(word_matches))

    # Stem matches (prefix of token), but skip tokens that overlap a word match
    for (_w, st) in query_terms:
        if not st:
            continue
        stem_lower = st.lower()
        for (tok, (s, e)) in zip(tokens, token_spans):
            tok_lower = tok.lower()
            if not tok_lower.startswith(stem_lower):
                continue
            span = (s, e)
            if span in word_matches or overlaps_any(span, word_matches):
                continue
            stem_matches.append(span)

    # Normalize stem matches
    stem_matches = sorted(set(stem_matches))

    # Base score: word=2, stem=1
    base_score = 2 * len(word_matches) + 1 * len(stem_matches)

    # Proximity bonus: +1 for each pair of matches within 50 chars
    all_spans = word_matches + stem_matches

    def spans_within_50(p: Tuple[int, int], q: Tuple[int, int]) -> bool:
        a, b = p
        c, d = q
        # Overlap?
        if not (b <= c or d <= a):
            return True
        # Non-overlap distance between edges
        dist = min(abs(a - d), abs(c - b))
        return dist <= 50

    bonus = 0
    for i in range(len(all_spans)):
        for j in range(i + 1, len(all_spans)):
            if spans_within_50(all_spans[i], all_spans[j]):
                bonus += 1

    total = base_score + bonus
    return (total, all_spans)

#---------------------------------- HIGHLIGHT KEYWORDS -------------------
def highlight_keywords(text: str,
                       spans: list[tuple[int, int]],
                       pattern: str = "**{}**") -> str:
    """
    Highlight substrings of `text` given by (start, end) character spans using `pattern.format(...)`.
    """
    if not text or not spans:
        return text

    n = len(text)

    # Normalize & sanitize spans: clamp to [0, n], discard empty/invalid
    norm = []
    for s, e in spans:
        if s is None or e is None:
            continue
        s = max(0, min(n, int(s)))
        e = max(0, min(n, int(e)))
        if e > s:  # non-empty
            norm.append((s, e))
    if not norm:
        return text

    # Sort by start, then end; merge overlapping/adjacent spans
    norm.sort()
    merged = []
    cs, ce = norm[0]
    for s, e in norm[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))

    # Rebuild the string with highlights applied
    out_parts = []
    prev = 0
    for s, e in merged:
        if prev < s:
            out_parts.append(text[prev:s])
        segment = text[s:e]
        try:
            out_parts.append(pattern.format(segment))
        except Exception:
            out_parts.append(f"{pattern}{segment}{pattern}")
        prev = e
    if prev < n:
        out_parts.append(text[prev:])

    return "".join(out_parts)

#--------------------------------------- SNIPPET GENERATOR ---------------
def snippet(query: str, filename: str, max_length: int = 250) -> str:
    """
    Build a snippet for `filename` given `query`, up to `max_length` characters.
    Returns HTML-safe snippet (with <b> tags).
    """
    if not query or not filename or max_length <= 0:
        return ""

    # load and normalize document text
    try:
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    except Exception:
        return ""

    # Strip HTML to plain text via BeautifulSoup (no regex fallback)
    try:
        text = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    except Exception:
        text = raw

    # Collapse whitespace
    text = " ".join(text.split())
    if not text:
        return ""

    # Sentence splitting
    try:
        sentences = sent_tokenize(text)
        offsets: List[Tuple[int, int]] = []
        pos = 0
        for s in sentences:
            start = text.find(s, pos)
            if start == -1:
                start = text.find(s)
            end = start + len(s)
            offsets.append((start, end))
            pos = end
    except Exception:
        # Fallback: single sentence (whole text)
        sentences = [text]
        offsets = [(0, len(text))]

    if not sentences:
        return ""

    # Preprocess query into (word, stem) tuples
    try:
        stops = set(stopwords.words("english"))
        lemm = WordNetLemmatizer()
        stemmer = PorterStemmer()

        toks = word_tokenize(query.lower())
        words = [t for t in toks if any(c.isalnum() for c in t) and t not in stops]

        # Lemmatize and dedup in order
        uniq = []
        seen = set()
        for w in words:
            lw = lemm.lemmatize(w)
            if lw and lw not in seen:
                seen.add(lw)
                uniq.append(lw)

        query_terms: List[Tuple[str, str]] = [(w, stemmer.stem(w)) for w in uniq]
    except Exception:
        # crude token split fallback (no regex)
        basic_tokens = []
        current = []
        for ch in query.lower():
            if ch.isalnum():
                current.append(ch)
            else:
                if current:
                    basic_tokens.append("".join(current))
                    current = []
        if current:
            basic_tokens.append("".join(current))

        unique_basic = []
        seen = set()
        for w in basic_tokens:
            if len(w) > 1 and w not in seen:
                seen.add(w)
                unique_basic.append(w)

        query_terms = [(w, w) for w in unique_basic]

    if not query_terms:
        return ""

    # Score each sentence
    scored = []  # (score, idx, spans_in_sentence, sentence_text)
    for idx, s in enumerate(sentences):
        score, spans = score_sentence(query_terms, s)
        if score > 0 and spans:
            scored.append((score, idx, spans, s))

    # If nothing matched, return the first sentence clipped to max_length
    if not scored:
        out = sentences[0][:max_length]
        return out

    # Highlight sentences
    highlighted_by_idx = {}
    length_by_idx = {}
    for score, idx, spans, s in scored:
        h = highlight_keywords(s, spans, pattern="<b>{}</b>")
        highlighted_by_idx[idx] = h
        length_by_idx[idx] = len(h)

    # Sort candidates by score desc, then shorter sentence, then earlier idx
    scored.sort(key=lambda x: (-x[0], length_by_idx[x[1]], x[1]))

    selected: List[int] = []

    # Helper: compute assembled length with ellipses if needed
    def assembled_length_with(candidate_indices: List[int]) -> int:
        if not candidate_indices:
            return 0
        sorted_idxs = sorted(candidate_indices)
        total = 0
        prev = None
        for i in sorted_idxs:
            if prev is not None and i != prev + 1:
                total += 5  # " ... "
            total += length_by_idx.get(i, 0)
            prev = i
        return total

    # Greedy selection of best sentences that fit within max_length
    for score, idx, spans, s in scored:
        test = selected + [idx]
        if assembled_length_with(test) <= max_length:
            selected = test

    if not selected:
        # if still empty due to very long single sentences, take best that fits
        for score, idx, spans, s in scored:
            if length_by_idx[idx] <= max_length:
                selected = [idx]
                break
        if not selected:
            # fall back to clipped best matching sentence
            best = max(scored, key=lambda x: x[0])
            h = highlighted_by_idx[best[1]]
            return h[:max_length]

    # Assemble final snippet in source order with ellipses
    selected.sort()
    parts: List[str] = []
    prev = None
    for i in selected:
        if prev is not None and i != prev + 1:
            parts.append(" ... ")
        parts.append(highlighted_by_idx[i])
        prev = i

    snippet_text = "".join(parts)

    # Trim at a whitespace boundary without regex
    if len(snippet_text) > max_length:
        cut = max_length
        # move back to the last whitespace before cut
        while cut > 0 and not snippet_text[cut - 1].isspace():
            cut -= 1
        if cut == 0:
            cut = max_length
        snippet_text = snippet_text[:cut].rstrip()

    return snippet_text