#Author: Caleb Corolewski
#Date: Fall 2025
import os
import sqlite3
from contextlib import closing
#---------------------- CREATE SQLite DATABASE --------------------------
def create_db(db_path: str = "index.db", reset: bool = True) -> None:
    """
    Create (or reset) the inverted index database.
    Tables:
        documents(doc_id, filename)
        terms(term_id, term, df, cf)
        postings(term_id, doc_id, tf)
        context_vectors(
    """
    # Delete existing DB if requested
    if reset and os.path.exists(db_path):
        os.remove(db_path)

    with closing(sqlite3.connect(db_path)) as conn, conn:  
        conn.execute("PRAGMA foreign_keys = ON;")

        # --- Table 1: documents ---
        conn.execute("""
            CREATE TABLE documents (
                doc_id   INTEGER PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL
            );
        """)

        # --- Table 2: terms ---
        conn.execute("""
            CREATE TABLE terms (
                term_id INTEGER PRIMARY KEY,
                term    TEXT UNIQUE NOT NULL,
                df      INTEGER NOT NULL DEFAULT 0,
                cf      INTEGER NOT NULL DEFAULT 0
            );
        """)

        # --- Table 3: postings ---
        conn.execute("""
            CREATE TABLE postings (
                term_id INTEGER NOT NULL,
                doc_id  INTEGER NOT NULL,
                tf      INTEGER NOT NULL,
                PRIMARY KEY (term_id, doc_id),
                FOREIGN KEY (term_id) REFERENCES terms(term_id) ON DELETE CASCADE,
                FOREIGN KEY (doc_id)  REFERENCES documents(doc_id) ON DELETE CASCADE
            );
        """)

        #--- Table 4 Query Likelihood
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_likelihood (
                doc_id   INTEGER NOT NULL,
                term_id  INTEGER NOT NULL,
                log_prob REAL    NOT NULL,
                PRIMARY KEY (doc_id, term_id),
                FOREIGN KEY (doc_id)  REFERENCES documents(doc_id) ON DELETE CASCADE,
                FOREIGN KEY (term_id) REFERENCES terms(term_id)    ON DELETE CASCADE
            );
        """)

        #indexes for speed
        conn.execute("CREATE INDEX idx_terms_term ON terms(term);")
        conn.execute("CREATE INDEX idx_postings_term ON postings(term_id);")
        conn.execute("CREATE INDEX idx_postings_doc  ON postings(doc_id);")
        conn.execute("CREATE INDEX idx_ql_doc       ON query_likelihood(doc_id);")
        conn.execute("CREATE INDEX idx_ql_term      ON query_likelihood(term_id);")

    print(f"[create_db] Database {db_path} created successfully.")

#------------------------------ Connect to DB -------------------------
def _connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn