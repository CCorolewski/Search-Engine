#Author: Caleb Corolewski
#Date: Fall 2025
import sqlite3
from contextlib import closing
from preprocessing import standard_preprocess
from database import _connect
#----------------------------- RANK BOOLEAN -------------------------------
def rank_boolean(
    query: str,
    db_path: str = "index.db",
    k: int = 10,
) -> list[tuple[float, str]]:
    """
    Rank documents using a Boolean-style similarity:
    """

    if not query or not isinstance(query, str):
        return []

    # Use the same analyzer as for BM25 / indexing
    q_terms = standard_preprocess(query)
    if not q_terms:
        return []

    # Distinct query terms
    q_unique = sorted(set(q_terms))

    with closing(_connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row

        # Map query terms -> term_id
        qmarks = ",".join("?" for _ in q_unique)
        term_rows = conn.execute(
            f"""
            SELECT term, term_id
            FROM terms
            WHERE term IN ({qmarks})
            """,
            q_unique,
        ).fetchall()

        if not term_rows:
            return []

        term_to_id: dict[str, int] = {
            r["term"]: int(r["term_id"]) for r in term_rows
        }

        # Only keep terms that exist in vocab
        vocab_terms = [t for t in q_unique if t in term_to_id]
        if not vocab_terms:
            return []

        term_ids = [term_to_id[t] for t in vocab_terms]

        # Get postings for these term_ids (presence/absence only)
        tmarks = ",".join("?" for _ in term_ids)
        post_rows = conn.execute(
            f"""
            SELECT term_id, doc_id
            FROM postings
            WHERE term_id IN ({tmarks})
            """,
            term_ids,
        ).fetchall()

        if not post_rows:
            return []

        # For each doc, track which query term_ids it contains
        doc_to_terms: dict[int, set[int]] = {}

        for r in post_rows:
            tid = int(r["term_id"])
            doc_id = int(r["doc_id"])
            s = doc_to_terms.setdefault(doc_id, set())
            s.add(tid)

        if not doc_to_terms:
            return []

        # Compute Boolean-style scores: number of matched query terms
        doc_scores: list[tuple[float, int]] = []
        for doc_id, term_set in doc_to_terms.items():
            score = float(len(term_set))  # coordination factor
            doc_scores.append((score, doc_id))

        # Map doc_id -> filename
        doc_ids = [doc_id for _, doc_id in doc_scores]
        dmarks = ",".join("?" for _ in doc_ids)
        fname_rows = conn.execute(
            f"""
            SELECT doc_id, filename
            FROM documents
            WHERE doc_id IN ({dmarks})
            """,
            doc_ids,
        ).fetchall()

        doc_to_name = {int(r["doc_id"]): r["filename"] for r in fname_rows}

        # Build final ranked list
        ranked: list[tuple[float, str]] = []
        for score, doc_id in doc_scores:
            fname = doc_to_name.get(doc_id)
            if fname is not None:
                ranked.append((score, fname))

        ranked.sort(key=lambda x: (-x[0], x[1]))
        return ranked[:k]
