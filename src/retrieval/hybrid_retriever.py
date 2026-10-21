from __future__ import annotations

import argparse
import logging
import re
from typing import Any, Dict, List, Optional

from src.indexing.db_client import PostgresVectorClient
from src.indexing.embedder import OllamaEmbedder

logger = logging.getLogger(__name__)

# Conversational words that Postgres' English stopword list keeps but that carry no
# topical meaning in a question. Postgres already drops "what", "how", "the", "is", ...
_QUERY_FILLER = {
    "explain", "describe", "tell", "please", "show", "give", "summarize", "summarise",
    "can", "could", "would", "me", "us", "kindly",
}
_TOKEN_RE = re.compile(r"[\w\-\.]+", re.UNICODE)


def build_keyword_query(question: str) -> str:
    """
    Turn a natural-language question into text for the keyword (sparse) side.
    Drops conversational filler; keeps identifiers like 'arXiv:2407.01528v3', 'GPT-4', 'Table 2'.
    Falls back to the original question if everything would be stripped.
    """
    tokens = _TOKEN_RE.findall(question)
    kept = [t for t in tokens if t.lower() not in _QUERY_FILLER]
    return " ".join(kept) if kept else question


class HybridRetriever:
    """
    Stage 1 of retrieval: BM25-style keyword search + semantic search, fused with RRF.
    The query is embedded with the same Ollama model used at indexing time (nomic-embed-text),
    otherwise query and chunk vectors would live in different spaces.
    """

    def __init__(
        self,
        embedder: Optional[OllamaEmbedder] = None,
        db_client: Optional[PostgresVectorClient] = None,
    ) -> None:
        self.embedder = embedder or OllamaEmbedder()
        self.db_client = db_client or PostgresVectorClient()

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_pool: int = 50,
        doc_name: Optional[str] = None,
        parent_title: Optional[str] = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        query_vector = self.embedder.embed_text(query)
        keyword_query = build_keyword_query(query)
        logger.debug("Hybrid search | keywords=%r | top_k=%d pool=%d", keyword_query, top_k, candidate_pool)

        return self.db_client.hybrid_search(
            query_text=keyword_query,
            query_vector=query_vector,
            top_k=top_k,
            candidate_pool=candidate_pool,
            doc_name=doc_name,
            parent_title=parent_title,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run a hybrid (BM25-style + semantic) search.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pool", type=int, default=50)
    parser.add_argument("--doc", default=None, help="Restrict to one doc_name")
    args = parser.parse_args()

    results = HybridRetriever().search(args.query, top_k=args.top_k, candidate_pool=args.pool, doc_name=args.doc)
    for i, r in enumerate(results, 1):
        sim = f"{r['similarity']:.3f}" if r["similarity"] is not None else "  -  "
        kw = f"{r['keyword_score']:.3f}" if r["keyword_score"] is not None else "  -  "
        print(
            f"{i}. rrf={r['rrf_score']:.4f} [{r['source']:6}] "
            f"dense#{r['dense_rank'] or '-'} sim={sim} | sparse#{r['sparse_rank'] or '-'} kw={kw}"
        )
        print(f"   {r['breadcrumb']}  (p.{r['page']})")
        print(f"   {r['text'][:160].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    _main()
