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

#------------------------------ BM25 RANKING ---------------------------------
def bm25_rank(
    query: str,
    db_path: str = "index.db",
    k: int = 10,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[tuple[float, str]]:
    """
    Rank documents using BM25, reading all stats from the SQLite index.

    Returns a list of (score, filename) sorted by descending score.
    """

    # 1. Preprocess query with SAME pipeline as documents
    terms = standard_preprocess(query)
    if not terms:
        return []

    unique_terms = sorted(set(terms))

    with closing(_connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row

        # 2. Get N = number of indexed documents
        row = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        N = int(row["n"]) if row is not None else 0
        if N == 0:
            return []

        # 3. Get document lengths (dl) and avdl from postings
        dl_rows = conn.execute(
            "SELECT doc_id, SUM(tf) AS dl FROM postings GROUP BY doc_id"
        ).fetchall()

        if not dl_rows:
            return []

        doc_len: dict[int, float] = {
            int(r["doc_id"]): float(r["dl"]) for r in dl_rows
        }
        avdl = sum(doc_len.values()) / len(doc_len)

        # 4. Map query terms -> (term_id, df)
        placeholders = ",".join("?" for _ in unique_terms)
        term_rows = conn.execute(
            f"""
            SELECT term, term_id, df
            FROM terms
            WHERE term IN ({placeholders})
            """,
            unique_terms,
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

        # Only keep query terms that are actually in the vocab
        term_ids = [term_to_id[t] for t in unique_terms if t in term_to_id]
        if not term_ids:
            return []

        # 5. Get postings for all query terms
        placeholders = ",".join("?" for _ in term_ids)
        post_rows = conn.execute(
            f"""
            SELECT term_id, doc_id, tf
            FROM postings
            WHERE term_id IN ({placeholders})
            """,
            term_ids,
        ).fetchall()

        if not post_rows:
            return []

        # 6. Accumulate BM25 scores by doc_id
        scores: dict[int, float] = defaultdict(float)

        for r in post_rows:
            term_id = int(r["term_id"])
            doc_id = int(r["doc_id"])
            f_td = float(r["tf"])
            df = float(term_df.get(term_id, 0))

            dl = doc_len.get(doc_id, 0.0)
            if dl <= 0.0 or df <= 0.0:
                continue

            # idf component
            idf = math.log((N - df + 0.5) / (df + 0.5))

            # K(d)
            K = _bm25_K(dl, avdl, k1, b)

            # tf component
            tf_weight = ((k1 + 1.0) * f_td) / (K + f_td)

            scores[doc_id] += idf * tf_weight

        if not scores:
            return []

        # 7. Map doc_ids to filenames
        top_doc_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)[:k]
        placeholders = ",".join("?" for _ in top_doc_ids)

        doc_rows = conn.execute(
            f"""
            SELECT doc_id, filename
            FROM documents
            WHERE doc_id IN ({placeholders})
            """,
            top_doc_ids,
        ).fetchall()

        id_to_name = {
            int(r["doc_id"]): r["filename"] for r in doc_rows
        }

        # 8. Build final ranked list (score, filename)
        results: list[tuple[float, str]] = []
        for doc_id in top_doc_ids:
            fname = id_to_name.get(doc_id)
            if fname is not None:
                results.append((scores[doc_id], fname))

        # sort again in case DB order scrambled them
        results.sort(key=lambda x: x[0], reverse=True)
        return results
