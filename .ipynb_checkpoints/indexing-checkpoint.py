#Author: Caleb Corolewski
#Date: Fall 2026
import os
from pathlib import Path
from contextlib import closing
import traceback
from collections import Counter
import sqlite3
import math
from preprocecssing import standard_preprocess, ql_preprocess
from database import _connect 
# --------------------- INDEX FILES -----------------------------
def index(filename: str, db_path: str = "index.db") -> None:
    """
    Index a single file using the same standard preprocessing used by
    get_document_vector() and index_vector_model().
    """

    # Read file
    try:
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    except Exception:
        return

    # Preprocess using unified pipeline
    terms = standard_preprocess(raw)

    # Convert to relative path
    stored_name = Path(os.path.relpath(filename)).as_posix()

    # Compute term frequencies
    from collections import Counter
    tf = Counter(terms)

    with closing(_connect(db_path)) as conn, conn:

        # Skip if already indexed
        row = conn.execute(
            "SELECT doc_id FROM documents WHERE filename = ?",
            (stored_name,)
        ).fetchone()
        if row is not None:
            return

        # Insert doc row
        cur = conn.execute(
            "INSERT INTO documents(filename) VALUES(?)",
            (stored_name,)
        )
        doc_id = cur.lastrowid

        # Empty document? Nothing else to do.
        if not tf:
            return

        # Insert terms into vocabulary
        conn.executemany(
            "INSERT OR IGNORE INTO terms(term) VALUES(?)",
            [(t,) for t in tf]
        )

        # Get term_ids
        qmarks = ",".join("?" for _ in tf)
        rows = conn.execute(
            f"SELECT term, term_id FROM terms WHERE term IN ({qmarks})",
            list(tf.keys())
        ).fetchall()
        term_to_id = {t: tid for (t, tid) in rows}

        # Update df and cf
        conn.executemany(
            "UPDATE terms SET df = df + 1, cf = cf + ? WHERE term_id = ?",
            [(tf[t], term_to_id[t]) for t in tf]
        )

        # Insert postings
        conn.executemany(
            "INSERT INTO postings(term_id, doc_id, tf) VALUES (?, ?, ?)",
            [(term_to_id[t], doc_id, tf[t]) for t in tf]
        )

#-----------------------------INDEXING A DIRECTORY--------------------------------
def index_dir(dirname: str, db_path: str = "index.db") -> None:
    if not os.path.isdir(dirname):
        raise NotADirectoryError(dirname)

    for root, dirs, files in os.walk(dirname):
        # skip hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for fname in sorted(files):
            fullpath = os.path.join(root, fname)
            if os.path.isfile(fullpath):
                try:
                    index(fullpath, db_path=db_path)
                except Exception as e:
                    print(f"[index_dir] Skipping {fullpath}: {e}")
                    traceback.print_exc()

#--------------------------- GET DOCUMENT VECTOR --------------------------
def get_document_vector(filename: str, db_path: str = "index.db") -> list[tuple[str, float]]:
    """
    Return the weighted document vector for the given filename as a list of (term, weight) tuples sorted by decreasing weight (then term for ties).
    """
    if not filename or not isinstance(filename, str):
        return []

    stored_name = Path(os.path.relpath(filename)).as_posix()

    with closing(_connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row

        # find doc_id for this filename
        row = conn.execute(
            "SELECT doc_id FROM documents WHERE filename = ?",
            (stored_name,)
        ).fetchone()

        if row is None:
            return []

        doc_id = int(row["doc_id"])

        rows = conn.execute("""
            SELECT term, weight
            FROM doc_vectors
            WHERE doc_id = ?
            ORDER BY weight DESC, term ASC
        """, (doc_id,)).fetchall()

        return [(r["term"], float(r["weight"])) for r in rows]
        
#-------------------------- INDEX QUERY LIKELIHOOD ----------------------
QL_MU = 2000.0  

def index_query_likelihood(db_path: str = "index.db") -> None:
    """
    Build Query Likelihood statistics by rescanning the raw documents
    with ql_preprocess(), which KEEPS stopwords so that |C| and |D|
    count all term occurrences.
    """

    with closing(_connect(db_path)) as conn, conn:
        conn.row_factory = sqlite3.Row

        # ensure core tables exist
        have = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        for needed in ("documents", "terms", "query_likelihood"):
            if needed not in have:
                raise RuntimeError(
                    f"Required table '{needed}' not found in {db_path}. "
                    "Run create_db() and index_dir() first."
                )

        # First pass: rescan docs with ql_preprocess
        doc_tf_ql: dict[int, Counter] = {}
        doc_len_ql: dict[int, int] = {}
        coll_tf_ql: Counter = Counter()

        doc_rows = conn.execute("SELECT doc_id, filename FROM documents").fetchall()
        if not doc_rows:
            return

        for row in doc_rows:
            doc_id = int(row["doc_id"])
            filename = row["filename"]

            try:
                with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
            except OSError:
                # Skip unreadable files but continue with others
                continue

            terms = ql_preprocess(raw)
            if not terms:
                continue

            tf = Counter(terms)
            doc_tf_ql[doc_id] = tf
            doc_len_ql[doc_id] = sum(tf.values())
            coll_tf_ql.update(tf)

        if not doc_tf_ql or not coll_tf_ql:
            return

        # |C| = total term occurrences in collection under QL preprocessing
        C = float(sum(coll_tf_ql.values()))
        if C <= 0.0:
            return

        # map QL terms to term_id using existing vocabulary
        vocab_terms = list(coll_tf_ql.keys())
        CHUNK = 800  # avoid huge IN() lists
        term_to_id: dict[str, int] = {}

        for i in range(0, len(vocab_terms), CHUNK):
            chunk = vocab_terms[i:i + CHUNK]
            qmarks = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT term, term_id FROM terms WHERE term IN ({qmarks})",
                chunk
            ).fetchall()
            term_to_id.update({r["term"]: int(r["term_id"]) for r in rows})

        # Clear old QL stats; schema already exists from create_db()
        conn.execute("DELETE FROM query_likelihood;")

        # Second pass: compute log p(t|D) for each doc & term and store
        for doc_id, tf in doc_tf_ql.items():
            D_len = float(doc_len_ql.get(doc_id, 0))
            denom = D_len + QL_MU
            if denom <= 0.0:
                continue

            data = []
            for term, f_td in tf.items():
                # only store for terms that exist in the main index vocab
                tid = term_to_id.get(term)
                if tid is None:
                    continue

                c_t = float(coll_tf_ql[term])  # QL collection frequency

                p_td = (float(f_td) + QL_MU * (c_t / C)) / denom
                if p_td <= 0.0:
                    continue

                log_p = math.log(p_td)
                data.append((doc_id, tid, log_p))

            if data:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO query_likelihood(doc_id, term_id, log_prob)
                    VALUES (?,?,?)
                    """,
                    data
                )
