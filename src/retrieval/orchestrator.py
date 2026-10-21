from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

import ollama

from src.indexing.db_client import PostgresVectorClient
from src.retrieval.evaluator import CitationAudit, EvaluationResult, RAGJudge, audit_citations
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.prompts import (
    DEFAULT_RAG_SYSTEM_PROMPT,
    build_rag_prompt_messages,
    build_rag_user_prompt,
)
from src.retrieval.reranker import CrossEncoderReranker, MxbaiReranker

logger = logging.getLogger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """Protocol defining the interface for any reranking model."""

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        text_field: str = "text",
    ) -> List[Dict[str, Any]]:
        ...


@dataclass
class RetrievalConfig:
    """Configuration options for the retrieval orchestrator."""
    top_k: int = 5
    candidate_pool: int = 30
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    min_rerank_score: Optional[float] = None
    expand_window: int = 0
    doc_name: Optional[str] = None
    parent_title: Optional[str] = None
    reorder_lost_in_middle: bool = True
    system_prompt: Optional[str] = None
    evaluate: bool = False
    judge_model: Optional[str] = None


def reorder_lost_in_middle(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Mitigate the 'Lost in the Middle' attention degradation effect in LLMs.
    
    Stanford research ('Lost in the Middle: How Language Models Use Long Contexts')
    shows that LLMs attend best to information at the very beginning (primacy)
    and very end (recency) of the prompt context, while struggling with information
    in the middle (a U-shaped attention curve).
    
    This function reorders ranked chunks so that:
      - Rank 1 (best) is placed at the BEGINNING.
      - Rank 2 (second best) is placed at the END.
      - Rank 3 is placed next to the beginning, Rank 4 next to the end...
      - The least relevant chunks are buried in the DEAD MIDDLE.
    """
    if len(chunks) <= 2:
        return list(chunks)

    left: List[Dict[str, Any]] = []
    right: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            # Even indices (0=Rank 1, 2=Rank 3, ...) placed on the left (beginning)
            left.append(chunk)
        else:
            # Odd indices (1=Rank 2, 3=Rank 4, ...) placed on the right (end)
            right.insert(0, chunk)

    return left + right


@dataclass
class RetrievalResult:
    """Structured response returned by the retrieval orchestrator."""
    query: str
    chunks: List[Dict[str, Any]]
    formatted_context: str
    metrics: Dict[str, Any] = field(default_factory=dict)


class RetrievalOrchestrator:
    """
    Production-grade Retrieval Orchestrator for RAG.
    
    Coordinates the full search lifecycle:
      1. Stage 1: Dual Dense (HNSW) + Sparse (GIN) search with Reciprocal Rank Fusion.
      2. Stage 2: Second-stage high-precision reranking using Cross-Encoder.
      3. Post-Processing: Score thresholding and context window expansion.
      4. Context Synthesis: Formats structured context blocks with section breadcrumbs and tables for the LLM.
      5. Optional Generation: Answers queries directly with local LLMs (e.g. llama3.2).
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
        db_client: Optional[PostgresVectorClient] = None,
        judge: Optional[RAGJudge] = None,
    ) -> None:
        self.stage1 = hybrid_retriever or HybridRetriever()
        self.stage2: Reranker = reranker or CrossEncoderReranker()
        self.db_client = db_client or PostgresVectorClient()
        self.judge = judge or RAGJudge()

    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
    ) -> RetrievalResult:
        """
        Execute end-to-end two-stage retrieval with context synthesis.
        """
        cfg = config or RetrievalConfig()
        t_total_start = time.perf_counter()

        if not query or not query.strip():
            return RetrievalResult(
                query=query,
                chunks=[],
                formatted_context="",
                metrics={"total_time_ms": 0.0, "stage1_time_ms": 0.0, "rerank_time_ms": 0.0},
            )

        # -------------------------------------------------------------
        # Step 1: Hybrid Retrieval (HNSW + GIN + RRF in PostgreSQL)
        # -------------------------------------------------------------
        t_s1_start = time.perf_counter()
        candidates = self.stage1.search(
            query=query,
            top_k=cfg.candidate_pool,
            candidate_pool=cfg.candidate_pool,
            doc_name=cfg.doc_name,
            parent_title=cfg.parent_title,
            dense_weight=cfg.dense_weight,
            sparse_weight=cfg.sparse_weight,
        )
        stage1_time_ms = (time.perf_counter() - t_s1_start) * 1000

        if not candidates:
            total_time_ms = (time.perf_counter() - t_total_start) * 1000
            return RetrievalResult(
                query=query,
                chunks=[],
                formatted_context="",
                metrics={"total_time_ms": total_time_ms, "stage1_time_ms": stage1_time_ms, "rerank_time_ms": 0.0},
            )

        # -------------------------------------------------------------
        # Step 2: High-Precision Reranker (mxbai-embed-large 1024-dim)
        # -------------------------------------------------------------
        t_s2_start = time.perf_counter()
        reranked = self.stage2.rerank(
            query=query,
            candidates=candidates,
            top_k=cfg.top_k,
        )
        rerank_time_ms = (time.perf_counter() - t_s2_start) * 1000

        # Filter by minimum rerank score if requested
        if cfg.min_rerank_score is not None:
            reranked = [c for c in reranked if c.get("rerank_score", 0.0) >= cfg.min_rerank_score]

        # -------------------------------------------------------------
        # Step 3: Context Window Expansion (Optional)
        # -------------------------------------------------------------
        final_chunks: List[Dict[str, Any]] = []
        if cfg.expand_window > 0:
            for c in reranked:
                doc_name = c.get("doc_name")
                chunk_idx = c.get("chunk_index")
                if doc_name and chunk_idx is not None:
                    surrounding = self.db_client.get_surrounding_chunks(
                        doc_name=doc_name,
                        chunk_index=chunk_idx,
                        window=cfg.expand_window,
                    )
                    # Merge surrounding text into expanded text
                    expanded_texts = [s["text"] for s in surrounding if s.get("text")]
                    c_copy = dict(c)
                    c_copy["expanded_text"] = "\n\n".join(expanded_texts)
                    c_copy["expanded_window"] = cfg.expand_window
                    final_chunks.append(c_copy)
                else:
                    final_chunks.append(c)
        else:
            final_chunks = reranked

        # -------------------------------------------------------------
        # Step 4: Lost-in-the-Middle Reordering
        # -------------------------------------------------------------
        # Annotate original descending relevance rank (1-indexed)
        for rank, c in enumerate(final_chunks, 1):
            c["relevance_rank"] = rank

        reordered = cfg.reorder_lost_in_middle and len(final_chunks) > 2
        if reordered:
            final_chunks = reorder_lost_in_middle(final_chunks)

        # -------------------------------------------------------------
        # Step 5: Format Context for LLM Injection
        # -------------------------------------------------------------
        formatted_context = self.format_context(final_chunks)
        total_time_ms = (time.perf_counter() - t_total_start) * 1000

        metrics = {
            "total_time_ms": round(total_time_ms, 2),
            "stage1_time_ms": round(stage1_time_ms, 2),
            "rerank_time_ms": round(rerank_time_ms, 2),
            "candidates_retrieved": len(candidates),
            "final_chunks_count": len(final_chunks),
            "reordered_lost_in_middle": reordered,
        }

        return RetrievalResult(
            query=query,
            chunks=final_chunks,
            formatted_context=formatted_context,
            metrics=metrics,
        )

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Synthesizes chunks into a clean, numbered context block ready for an LLM prompt.
        Preserves section breadcrumbs, markdown tables, and citations.
        """
        if not chunks:
            return ""

        context_blocks: List[str] = []
        for i, c in enumerate(chunks, 1):
            doc = str(c.get("doc_name") or "Unknown Document")
            page = c.get("page") or 1
            breadcrumb = str(c.get("breadcrumb") or "")
            raw_score = c.get("rerank_score")
            if raw_score is None:
                raw_score = c.get("rrf_score")
            try:
                score = float(raw_score) if raw_score is not None else 0.0
            except (ValueError, TypeError):
                score = 0.0

            rank = c.get("relevance_rank")

            header = f"[Source {i}] Document: {doc} | Page: {page}"
            if breadcrumb:
                header += f" | Section: {breadcrumb}"
            if rank is not None:
                header += f" (Relevance Rank: #{rank}, Score: {score:.3f})"
            else:
                header += f" (Relevance: {score:.3f})"

            # Use expanded text if available, otherwise original text
            body = str(c.get("expanded_text") or c.get("text") or "")

            # Include extracted table markdown if present
            tbl = c.get("table_markdown")
            if tbl and str(tbl).strip() and str(tbl).strip() not in body:
                body += f"\n\nTable Data:\n{str(tbl).strip()}"

            context_blocks.append(f"{header}\n{body.strip()}")

        return "\n\n---\n\n".join(context_blocks)

    def build_prompt(
        self,
        query: str,
        retrieval_or_context: Union[RetrievalResult, str, List[Dict[str, Any]]],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Builds the structured prompt payload enforcing strict grounding,
        inline citations, conflict-handling rules, and chunk interpretation.
        """
        if isinstance(retrieval_or_context, RetrievalResult):
            formatted_context = retrieval_or_context.formatted_context
        elif isinstance(retrieval_or_context, list):
            formatted_context = self.format_context(retrieval_or_context)
        else:
            formatted_context = retrieval_or_context

        return build_rag_prompt_messages(
            query=query,
            formatted_context=formatted_context,
            system_prompt=system_prompt,
        )

    def evaluate_answer(
        self,
        query: str,
        answer: str,
        retrieval: RetrievalResult,
        synthesis_model: str = "llama3.2:latest",
        judge_model: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Standalone evaluation of a candidate answer using the anti-biased RAGJudge.
        """
        return self.judge.evaluate(
            query=query,
            formatted_context=retrieval.formatted_context,
            candidate_answer=answer,
            num_sources=len(retrieval.chunks),
            synthesis_model=synthesis_model,
            judge_model=judge_model,
        )

    def answer(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
        model: str = "llama3.2:latest",
        system_prompt: Optional[str] = None,
        evaluate: Optional[bool] = None,
        judge_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        End-to-End RAG: Retrieves context, formats grounded prompt with conflict
        rules and citation requirements, generates an answer using an LLM, and
        optionally runs anti-biased LLM-as-a-Judge evaluation.
        """
        retrieval = self.retrieve(query, config)

        if not retrieval.chunks:
            return {
                "answer": "No relevant documents were found to answer your question.",
                "retrieval": retrieval,
            }

        effective_system = system_prompt or (config.system_prompt if config else None) or DEFAULT_RAG_SYSTEM_PROMPT
        messages = self.build_prompt(
            query=query,
            retrieval_or_context=retrieval,
            system_prompt=effective_system,
        )

        t_llm_start = time.perf_counter()
        response: Any = ollama.chat(
            model=model,
            messages=messages,
        )
        llm_time_ms = (time.perf_counter() - t_llm_start) * 1000

        retrieval.metrics["llm_time_ms"] = round(llm_time_ms, 2)
        retrieval.metrics["total_with_llm_ms"] = round(
            retrieval.metrics["total_time_ms"] + llm_time_ms, 2
        )

        content: str = ""
        if hasattr(response, "message") and hasattr(response.message, "content"):
            content = str(response.message.content or "")
        elif isinstance(response, dict):
            msg = response.get("message", {})
            if isinstance(msg, dict):
                content = str(msg.get("content", ""))
            else:
                content = str(getattr(msg, "content", ""))

        result_dict: Dict[str, Any] = {
            "answer": content,
            "retrieval": retrieval,
        }

        should_eval = evaluate if evaluate is not None else (config.evaluate if config else False)
        if should_eval and content:
            active_judge_model = judge_model or (config.judge_model if config else None)
            eval_res = self.judge.evaluate(
                query=query,
                formatted_context=retrieval.formatted_context,
                candidate_answer=content,
                num_sources=len(retrieval.chunks),
                synthesis_model=model,
                judge_model=active_judge_model,
            )
            result_dict["evaluation"] = eval_res

        return result_dict


def _main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Orchestrator CLI")
    parser.add_argument("query", help="User question or search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final chunks")
    parser.add_argument("--pool", type=int, default=30, help="Stage 1 candidate pool")
    parser.add_argument("--doc", default=None, help="Filter by document name")
    parser.add_argument("--expand", type=int, default=0, help="Context window expansion size (number of neighboring chunks)")
    parser.add_argument("--answer", action="store_true", help="Generate answer using llama3.2")
    parser.add_argument("--evaluate", action="store_true", help="Run anti-biased LLM-as-a-Judge evaluation on generated answer")
    parser.add_argument("--judge-model", default="qwen2.5:latest", help="Judge model name (default: qwen2.5:latest to avoid same-model bias)")
    args = parser.parse_args()

    cfg = RetrievalConfig(
        top_k=args.top_k,
        candidate_pool=args.pool,
        doc_name=args.doc,
        expand_window=args.expand,
        evaluate=args.evaluate,
        judge_model=args.judge_model,
    )
    orch = RetrievalOrchestrator()

    if args.answer:
        answer_res = orch.answer(args.query, config=cfg)
        print("\n==================== LLM ANSWER ====================")
        print(answer_res["answer"])
        print("\n==================== RETRIEVAL METRICS ====================")
        ret_obj = answer_res["retrieval"]
        print(ret_obj.metrics if hasattr(ret_obj, "metrics") else {})

        if "evaluation" in answer_res:
            ev = answer_res["evaluation"]
            print("\n==================== LLM-AS-A-JUDGE EVALUATION ====================")
            print(f"Passed: {ev.passed} | Overall Score: {ev.overall_score} (Judge: {ev.judge_model})")
            print(f"Faithfulness: {ev.faithfulness_score} - {ev.faithfulness_reasoning}")
            print(f"Relevance: {ev.relevance_score} - {ev.relevance_reasoning}")
            print(f"Conflict Handling: {ev.conflict_handling_score} - {ev.conflict_handling_reasoning}")
            print(f"Citation Audit: {ev.citation_audit.valid_citations}/{ev.citation_audit.total_citations} valid (Invalid: {ev.citation_audit.invalid_citations})")
            if ev.same_model_bias_warning:
                print("WARNING: Same-model bias risk detected (generator and judge are identical).")
    else:
        retrieval_res = orch.retrieve(args.query, config=cfg)
        print(f"\nRetrieved {len(retrieval_res.chunks)} chunks in {retrieval_res.metrics['total_time_ms']} ms "
              f"(Stage 1: {retrieval_res.metrics['stage1_time_ms']} ms, Rerank: {retrieval_res.metrics['rerank_time_ms']} ms):\n")
        print(retrieval_res.formatted_context)



if __name__ == "__main__":
    _main()
