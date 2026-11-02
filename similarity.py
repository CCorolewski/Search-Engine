#Author: Caleb Corolewski
#Date: Fall 2025
from abc import ABC, abstractmethod
from BM25 import rank_bm25
from booleanranking import rank_boolean
from querylikelihood import rank_query_likelihood

#-----------------------------------SIMILARITY ABSTRACTION --------------------
class Similarity(ABC):
    """
    Lucene-style Similarity abstraction.
    A Similarity is responsible for ranking documents for a query.
    """

    @abstractmethod
    def rank(
        self,
        query: str,
        db_path: str = "index.db",
        k: int = 10,
    ) -> list[tuple[float, str]]:
        """
        Return a list of (score, filename) pairs sorted by descending score.
        """
        raise NotImplementedError

#-------------------------------- BM25 SIMILARITY ------------------------------
class BM25Similarity(Similarity):
    """
    BM25 ranking, wrapping your rank_bm25() function.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def rank(
        self,
        query: str,
        db_path: str = "index.db",
        k: int = 10,
    ) -> list[tuple[float, str]]:
        # assuming rank_bm25 signature: (query, db_path, k, k1, b)
        return rank_bm25(
            query=query,
            db_path=db_path,
            k=k,
            k1=self.k1,
            b=self.b,
        )

#------------------------------ QUERY LIKELIHOOD SIMILARITY ----------------
class QLSimilarity(Similarity):
    """
    Query Likelihood (Dirichlet-smoothed) ranking, using the query_likelihood table.
    """

    def rank(
        self,
        query: str,
        db_path: str = "index.db",
        k: int = 10,
    ) -> list[tuple[float, str]]:
        # rank_query_likelihood returns full list; we just slice top-k
        ranked = rank_query_likelihood(query, db_path=db_path)
        return ranked[:k]

#--------------------------------- BOOLEAN SIMILARITY ----------------------
class BooleanSimilarity(Similarity):
    """
    Boolean-style similarity:
    - Score = number of distinct query terms present in the document.
    - Uses OR semantics (documents with at least one matching term).
    """

    def rank(
        self,
        query: str,
        db_path: str = "index.db",
        k: int = 10,
    ) -> list[tuple[float, str]]:
        return rank_boolean(query=query, db_path=db_path, k=k)
