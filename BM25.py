#Author: Caleb Corolewski
#Date: Fall 2025
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from preprocessing import standard_preprocess, ql_preprocess
from database import _connect   

#----------------------------------- HELPER FUNCTION -------------------------
def _bm25_K(dl: float, avdl: float, k1: float, b: float) -> float:
    """
    BM25 normalization factor K.
    dl   = document length (sum of tf)
    avdl = average document length
    """
    return k1 * ((1.0 - b) + b * (dl / avdl))

#------------------------------- RANK BM25 --------------------------------
def rank_bm25(
    query: str,
    db_path: str = "index.db",
    k: int = 10,
    k1: float = 1.2,
    b: float = 0.75,
):
    """
    Rank documents using the BM25 model, using statistics from the SQLite index.
    """

    if not query or not isinstance(query, str):
        return []

    # Preprocess query with the same pipeline used for documents
    q_terms = standard_preprocess(query)
    if not q_terms:
        return []

    q_tf = Counter(q_terms)
    q_unique = list(q_tf.keys())

    with closing(_connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row

        # N = number of indexed documents
        row = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        N = int(row["n"]) if row is not None else 0
        if N == 0:
            return []

        # Document lengths dl = SUM(tf) per doc
        dl_rows = conn.execute(
            "SELECT doc_id, SUM(tf) AS dl FROM postings GROUP BY doc_id"
        ).fetchall()
        if not dl_rows:
            return []

        doc_len = {int(r["doc_id"]): float(r["dl"] or 0.0) for r in dl_rows}
        if not doc_len:
            return []

        avdl = sum(doc_len.values()) / len(doc_len)

        # Map query terms -> (term_id, df)
        qmarks = ",".join("?" for _ in q_unique)
        term_rows = conn.execute(
            f"""
            SELECT term, term_id, df
            FROM terms
            WHERE term IN ({qmarks})
            """,
            q_unique,
        ).fetchall()
        if not term_rows:
            return []

        term_to_id: dict[str, int] = {}
        term_df: dict[int, int] = {}
        for r in term_rows:
            t = r["term"]
            tid = int(r["term_id"])
            df = int(r["df"])
            term_to_id[t] = tid
            term_df[tid] = df

        # Only keep query terms that exist in the vocab
        vocab_terms = [t for t in q_unique if t in term_to_id]
        if not vocab_terms:
            return []

        term_ids = [term_to_id[t] for t in vocab_terms]
        id_to_term = {term_to_id[t]: t for t in vocab_terms}

        # Get postings for these term_ids
        tmarks = ",".join("?" for _ in term_ids)
        post_rows = conn.execute(
            f"""
            SELECT term_id, doc_id, tf
            FROM postings
            WHERE term_id IN ({tmarks})
            """,
            term_ids,
        ).fetchall()
        if not post_rows:
            return []

        # Accumulate BM25 scores per doc
        scores: dict[int, float] = defaultdict(float)

        for r in post_rows:
            term_id = int(r["term_id"])
            doc_id = int(r["doc_id"])
            f_td = float(r["tf"])

            dl = doc_len.get(doc_id, 0.0)
            df = float(term_df.get(term_id, 0))

            if dl <= 0.0 or df <= 0.0 or f_td <= 0.0:
                continue

            # idf component
            idf = math.log((N - df + 0.5) / (df + 0.5))

            # length normalization K
            K = _bm25_K(dl, avdl, k1, b)

            # term frequency component
            tf_weight = ((k1 + 1.0) * f_td) / (K + f_td)

            # optional query term frequency factor (q_tf)
            term = id_to_term.get(term_id)
            if term is None:
                continue
            qf = float(q_tf.get(term, 1))

            scores[doc_id] += qf * idf * tf_weight

        if not scores:
            return []

        # Map doc_id -> filename
        doc_ids = list(scores.keys())
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

        # Build ranked list (score, filename)
        ranked: list[tuple[float, str]] = []
        for doc_id, score in scores.items():
            fname = doc_to_name.get(doc_id)
            if fname is not None:
                ranked.append((score, fname))

        ranked.sort(key=lambda x: (-x[0], x[1]))
        # top-k
        return ranked[:k]

