#****************************************************************************
# webap.py - Utilities for working with the WebAP dataset
#
# Boise State University CS 437
# Dr. Henderson
# Version 1
#
#----------------------------------------------------------------------------
import os
from pathlib import Path

WEBAP_DIR = Path(__file__).parent

def webap_rel_key(query_id, k=None, rel_file=f"{WEBAP_DIR}/qrels.txt"):
    results = []
    with open(rel_file) as file:
        for line in file:
            (qid, name, rel) = line.split()
            if int(qid) == query_id:
                results.append((rel, name))
    results = [ name for _, name in sorted(results) ]
    return results[:k]

def webap_query(query_id, query_file=f"{WEBAP_DIR}/queries.txt"):
    with open(query_file) as file:
        for line in file:
            (qid, query) = line.split('|')
            if int(qid) == query_id:
                return (qid, query)

def webap_queries(query_file=f"{WEBAP_DIR}/queries.txt"):
    queries = []
    with open(query_file) as file:
        for line in file:
            qid, q = line.split('|')
            queries.append((int(qid), q))
    return queries


