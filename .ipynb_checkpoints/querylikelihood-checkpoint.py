# Author: Caleb Corolewski
# Date: Fall 2025

import math
import sqlite3
from collections import Counter
from contextlib import closing
from preprocessing import ql_preprocess
from database import _connect


# -------------------------- RANK QUERY LIKELIHOOD (AND) -----------------------
def rank_query_likelihood(query: str, db_path: str = "index.db"):
    """
    Rank documents using the Query Likelihood model, but ONLY return documents
    that contain *all* query terms (Boolean AND behavior).
    """

    if not query or not isinstance(query, str):
        return []

    # Preprocess query terms with the QL-specific pipeline
    q_terms = ql_preprocess(query)
    if not q_terms:
        return []

    # Query term frequencies
    q_tf = Counter(q_terms)

    with closing(_connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row

        # Map query terms -> term_id
        q_unique = list(q_tf.keys())
        qmarks = ",".join("?" for _ in q_unique)

        rows = conn.execute(
            f"""
            SELECT term, term_id
            FROM terms
            WHERE term IN ({qmarks})
            """,
            q_unique,
        ).fetchall()

        if not rows:
            return []

        term_to_id = {r["term"]: int(r["term_id"]) for r in rows}

        # Only keep query terms that exist in the vocab
        vocab_terms = [t for t in q_unique if t in term_to_id]
        if not vocab_terms:
            return []

        term_ids = [term_to_id[t] for t in vocab_terms]
        id_to_term = {term_to_id[t]: t for t in vocab_terms}
        required_term_ids = set(term_ids)

        # Get log P(t|D) for these term_ids
        tmarks = ",".join("?" for _ in term_ids)
        tf_rows = conn.execute(
            f"""
            SELECT doc_id, term_id, log_prob
            FROM query_likelihood
            WHERE term_id IN ({tmarks})
            """,
            term_ids,
        ).fetchall()

        if not tf_rows:
            return []

        # First pass: figure out which docs contain
        #    ALL query terms (Boolean AND filter)
        doc_term_ids: dict[int, set[int]] = {}
        for r in tf_rows:
            doc_id = int(r["doc_id"])
            term_id = int(r["term_id"])
            doc_term_ids.setdefault(doc_id, set()).add(term_id)

        # Only keep docs whose term_id set covers all query term_ids
        candidate_doc_ids = {
            doc_id
            for doc_id, tids in doc_term_ids.items()
            if required_term_ids.issubset(tids)
        }

        if not candidate_doc_ids:
            # No document contains all query terms
            return []

        # Second pass: accumulate log P(Q|D) *only*
        #    for docs that passed the AND filter
        log_scores: dict[int, float] = {}

        for r in tf_rows:
            doc_id = int(r["doc_id"])
            if doc_id not in candidate_doc_ids:
                continue  # enforce AND

            term_id = int(r["term_id"])
            log_p_td = float(r["log_prob"])

            term = id_to_term.get(term_id)
            if term is None:
                continue

            q_count = q_tf.get(term, 0)
            if q_count == 0:
                continue

            log_scores[doc_id] = log_scores.get(doc_id, 0.0) + q_count * log_p_td

        if not log_scores:
            return []

        # Map doc_id -> filename and build ranked list
        doc_ids = list(log_scores.keys())
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

        ranked = []
        for doc_id, log_score in log_scores.items():
            fname = doc_to_name.get(doc_id)
            if fname is None:
                continue
            ranked.append((log_score, fname))

        # Sort by log P(Q|D) descending, then filename
        ranked.sort(key=lambda x: (-x[0], x[1]))
        return ranked


# -------------------------- QUERY LIKELIHOOD (AND) ----------------------------
def query_likelihood(query: str, db_path: str = "index.db") -> list[str]:
    """
    Wrapper around rank_query_likelihood() that returns only filenames.
    """
    ranked = rank_query_likelihood(query, db_path=db_path)
    return [fname for score, fname in ranked]
