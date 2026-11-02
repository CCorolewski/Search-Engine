# build_index.py

import sys
import os

from database import create_db
from indexing import index_dir, index_query_likelihood

def build_index(corpus_dir: str, db_path: str = "index.db", reset: bool = True) -> None:
    """
    Build (or rebuild) the search index for all files under corpus_dir.
    """
    corpus_dir = os.path.abspath(corpus_dir)

    if not os.path.isdir(corpus_dir):
        raise NotADirectoryError(f"Not a directory: {corpus_dir}")

    print(f"[build_index] Using database: {db_path}")
    print(f"[build_index] Indexing directory: {corpus_dir}")
    print(f"[build_index] reset = {reset}")

    # 1) Create / reset DB
    create_db(db_path=db_path, reset=reset)

    # 2) Index all files under corpus_dir
    index_dir(corpus_dir, db_path=db_path)

    # 3) Build Query Likelihood stats (if you’re using QLSimilarity)
    index_query_likelihood(db_path=db_path)

    print("[build_index] Done.")


if __name__ == "__main__":
    # Usage:
    #   python build_index.py /path/to/corpus [index.db]
    if len(sys.argv) < 2:
        print("Usage: python build_index.py <corpus_dir> [db_path]")
        sys.exit(1)

    corpus_dir = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) >= 3 else "index.db"

    try:
        build_index(corpus_dir, db_path=db_path, reset=True)
    except Exception as e:
        print(f"[build_index] Error: {e}")
        sys.exit(1)
