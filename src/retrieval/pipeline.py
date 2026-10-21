from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List, Optional

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker, MxbaiReranker

logger = logging.getLogger(__name__)


class TwoStageRetriever:
    """
    Complete Two-Stage Advanced RAG Retrieval Pipeline:
      Stage 1: Fast Hybrid Search (pgvector HNSW cosine + GIN tsvector full-text search)
               using `nomic-embed-text` (768-dim) fused via Reciprocal Rank Fusion (RRF).
      Stage 2: High-Precision Reranking over candidates using Cross-Encoder token attention.
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Any] = None,
    ) -> None:
        self.stage1 = hybrid_retriever or HybridRetriever()
        self.stage2 = reranker or CrossEncoderReranker()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 30,
        doc_name: Optional[str] = None,
        parent_title: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute end-to-end two-stage retrieval.
        
        Args:
            query: User's question or search prompt.
            top_k: Number of final chunks to return for LLM context.
            candidate_pool: Number of Stage-1 candidates to retrieve before reranking.
            doc_name: Optional filter to restrict search to a specific document.
            parent_title: Optional filter for a specific section title.
            
        Returns:
            List of top_k reranked chunk dictionaries.
        """
        # Stage 1: Fast hybrid retrieval in PostgreSQL
        candidates = self.stage1.search(
            query=query,
            top_k=candidate_pool,
            candidate_pool=candidate_pool,
            doc_name=doc_name,
            parent_title=parent_title,
        )

        if not candidates:
            return []

        # Stage 2: Second-stage reranker with mxbai-embed-large
        reranked = self.stage2.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

        return reranked


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end Two-Stage Retrieval (Hybrid + Reranker).")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final results")
    parser.add_argument("--pool", type=int, default=30, help="Stage-1 candidate pool size")
    parser.add_argument("--doc", default=None, help="Filter by document name")
    args = parser.parse_args()

    pipeline = TwoStageRetriever()
    results = pipeline.retrieve(args.query, top_k=args.top_k, candidate_pool=args.pool, doc_name=args.doc)

    print(f"\nQuery: {args.query}")
    print(f"Retrieved {len(results)} reranked results (from {args.pool} candidates):\n")

    for i, r in enumerate(results, 1):
        print(f"[{i}] Rerank Score: {r['rerank_score']:.4f} (RRF: {r['rrf_score']:.4f}, Stage-1: {r['source']})")
        print(f"    Breadcrumb: {r['breadcrumb']} (Page {r['page']})")
        print(f"    Text: {r['text'][:180].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    _main()
