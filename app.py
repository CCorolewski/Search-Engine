# Author: Caleb Corolewski
# Date: Fall 2025

import os
from pathlib import Path
from flask import Flask, render_template, request
from search import Searcher, snippet
from similarity import BM25Similarity, QLSimilarity, BooleanSimilarity
from scoring import f_score, avg_precision, reciprocal_rank, normalize_docnames

app = Flask(__name__)
DB_PATH = "index.db"

# ---------------------- WebAP dataset config ---------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "datasets" / "webap"
WEBAP_QUERY_PATH = DATA_ROOT / "queries.txt"
WEBAP_QRELS_PATH = DATA_ROOT / "qrels.txt"

_webap_queries_cache = None
_webap_rels_cache = None


# ---------------------- WebAP helper functions -------------------- #

def load_webap_queries():
    """
    Load WebAP queries into a dict: qid -> query_text.
    """
    global _webap_queries_cache
    if _webap_queries_cache is not None:
        return _webap_queries_cache

    mapping = {}

    try:
        with open(WEBAP_QUERY_PATH, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                qid_str = None
                text = None
                qid_part, text_part = line.split("|", 1)
                qid_str = qid_part.strip()
                text = text_part.strip()

                if not qid_str or not text:
                    continue

                try:
                    qid = int(qid_str)
                except ValueError:
                    # Skip any weird header / non-numeric qid lines
                    continue

                mapping[qid] = text
    except OSError:
        # If the file is missing, just leave mapping empty
        mapping = {}

    _webap_queries_cache = mapping
    return mapping


def get_webap_qid_for_text(query_text: str):
    """
    Map the current free-text query to a WebAP qid, if possible.
    """
    if not query_text:
        return None

    queries = load_webap_queries()
    q_norm = query_text.strip().lower()

    for qid, text in queries.items():
        if text.strip().lower() == q_norm:
            return qid

    # No exact match found
    return None


def load_webap_rels():
    """
    Load WebAP relevance judgments into a dict: qid -> set(doc_id).
    """
    global _webap_rels_cache
    if _webap_rels_cache is not None:
        return _webap_rels_cache

    rels = {}

    try:
        with open(WEBAP_QRELS_PATH, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                # We expect at least "qid DOCID REL"
                if len(parts) < 3:
                    continue

                try:
                    qid = int(parts[0])
                except ValueError:
                    # Skip lines where the first token isn't an integer qid
                    continue

                # Two common formats:
                #  - TREC style: qid 0 DOCID REL   (len >= 4)
                #  - Simple:     qid DOCID REL     (len == 3)
                if len(parts) >= 4:
                    doc_id = parts[2]
                    rel_str = parts[3]
                else:
                    doc_id = parts[1]
                    rel_str = parts[2]

                try:
                    rel = int(rel_str)
                except ValueError:
                    continue

                # Only keep positively judged documents
                if rel > 0 and doc_id:
                    rels.setdefault(qid, set()).add(doc_id)
    except OSError:
        rels = {}

    _webap_rels_cache = rels
    return rels


def get_rels_for_qid(qid: int):
    """
    Return the set of relevant doc IDs for this WebAP qid.
    """
    rels = load_webap_rels()
    return rels.get(qid, set())


def filename_to_docid(path: str) -> str:
    """
    Convert an indexed filename to the doc ID used in WebAP qrels.
    """
    base = Path(path).name
    docid = Path(base).stem
    return docid


# ------------------------ Similarity selector ---------------------- #

def get_similarity(model_name: str):
    """
    Map a string to a Similarity implementation.
    """
    name = (model_name or "").lower()
    if name == "ql":
        return QLSimilarity()
    elif name == "bool":
        return BooleanSimilarity()
    # default to BM25
    return BM25Similarity()


# ----------------------------- Flask app --------------------------- #

@app.route("/", methods=["GET"])
def home():
    """
    Show search form (and results if query is provided), and
    per-query WebAP evaluation metrics if we can compute them.
    """
    query = request.args.get("q", "").strip()
    model = request.args.get("model", "bm25").lower()
    k_param = request.args.get("k", "10").strip()

    try:
        k = int(k_param) if k_param else 10
    except ValueError:
        k = 10

    results = []

    # Always provide a metrics dict so the template can render safely.
    metrics = {
        "qid": None,
        "num_rels": 0,
        "precision_at_k": None,
        "recall_at_k": None,
        "avg_precision": None,
        "reciprocal_rank": None,
        "k": k,
    }

    if query:
        similarity = get_similarity(model)
        searcher = Searcher(db_path=DB_PATH, similarity=similarity)

        # Retrieve top-k (score, filename) pairs
        hits = searcher.search(query, k=k)

        # Normalize filenames to doc IDs that match qrels.txt
        # (e.g., '.../GX237-99-8478409.txt' -> 'GX237-99-8478409')
        filenames = [fname for score, fname in hits]
        doc_ids = normalize_docnames(filenames)

        # Build results for display (keep scores + snippets)
        for (score, fname), doc_id in zip(hits, doc_ids):
            try:
                snip = snippet(query, fname, max_length=250)
            except Exception:
                snip = ""

            results.append(
                {
                    "score": score,
                    "filename": fname,
                    "doc_id": doc_id,
                    "snippet": snip,
                }
            )

        # --- WebAP evaluation: map query -> qid, then qid -> rel docs --- #
        qid = get_webap_qid_for_text(query)
        if qid is not None:
            rels = get_rels_for_qid(qid)
            if rels:
                # For Precision/Recall@k, use only the top-k docs
                k_eff = min(k, len(doc_ids)) if doc_ids else k
                top_k_docs = doc_ids[:k_eff]

                # f_score(docs, rels) -> (recall, precision, f1)
                recall_k, precision_k, _f1 = f_score(top_k_docs, rels)

                # AP and RR over the full retrieved ranking
                ap = avg_precision(doc_ids, rels)
                rr = reciprocal_rank(doc_ids, rels)

                metrics.update(
                    {
                        "qid": qid,
                        "num_rels": len(rels),
                        "precision_at_k": precision_k,
                        "recall_at_k": recall_k,
                        "avg_precision": ap,
                        "reciprocal_rank": rr,
                        "k": k_eff,
                    }
                )

    return render_template(
        "search.html",
        query=query,
        model=model,
        k=k,
        results=results,
        metrics=metrics,
    )


if __name__ == "__main__":
    app.run(debug=True)
