#****************************************************************************
# scoring.py - Utilities for scoring IR engines
#
# Boise State University CS 437
# Dr. Henderson
# Version 1
#
# Computes various scores given a list of relevant docs and results:
# docs - a list normalized document names retrieved by the search engine
# rels - a list of relevant documents for comparison
#
# Note: Make sure to use normalize_docs() on your results before using
#       these functions
#----------------------------------------------------------------------------
import os
from pathlib import Path

def f_score(docs, rels):
    if not rels: return (0, 0, 0)
    reldocs = sum([ 1 if f in rels else 0 for f in docs ])
    if reldocs == 0: return (0, 0, 0)
    recall =  reldocs / len(rels)
    precision = reldocs / len(docs)
    f1 = 2 * precision * recall / (precision + recall)
    return (recall, precision, f1)

def precision_at(rank, docs, rels):
    reldocs = sum([ 1 if f in rels else 0 for f in docs[:rank]])
    precision = reldocs / rank
    return precision

def precisions(docs, rels):
    ps = []
    for i in range(0,len(docs)):
        if docs[i] in rels:
            ps.append(precision_at(i+1, docs, rels))
    return ps

def avg_precision(docs, rels):
    ps = precisions(docs, rels)
    return sum(ps) / len(ps) if len(ps) > 0 else 0

def reciprocal_rank(docs, rels):
    for i in range(len(docs)):
        if docs[i] in rels: return 1/(i+1)
    return 0

def normalize_docnames(docs):
    return [ Path(f).stem for f in docs ]

    