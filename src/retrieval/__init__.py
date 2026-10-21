from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.hybrid_retriever import HybridRetriever, build_keyword_query
    from src.retrieval.reranker import MxbaiReranker
    from src.retrieval.pipeline import TwoStageRetriever
    from src.retrieval.orchestrator import (
        RetrievalConfig,
        RetrievalResult,
        RetrievalOrchestrator,
    )
    from src.retrieval.prompts import (
        DEFAULT_RAG_SYSTEM_PROMPT,
        build_rag_user_prompt,
        build_rag_prompt_messages,
    )
    from src.retrieval.evaluator import (
        CitationAudit,
        EvaluationResult,
        RAGJudge,
        audit_citations,
    )

__all__ = [
    "HybridRetriever",
    "build_keyword_query",
    "MxbaiReranker",
    "TwoStageRetriever",
    "RetrievalConfig",
    "RetrievalResult",
    "RetrievalOrchestrator",
    "DEFAULT_RAG_SYSTEM_PROMPT",
    "build_rag_user_prompt",
    "build_rag_prompt_messages",
    "CitationAudit",
    "EvaluationResult",
    "RAGJudge",
    "audit_citations",
]


def __getattr__(name: str):
    if name in ("HybridRetriever", "build_keyword_query"):
        from src.retrieval.hybrid_retriever import HybridRetriever, build_keyword_query
        return locals()[name]
    if name == "MxbaiReranker":
        from src.retrieval.reranker import MxbaiReranker
        return MxbaiReranker
    if name == "TwoStageRetriever":
        from src.retrieval.pipeline import TwoStageRetriever
        return TwoStageRetriever
    if name in ("RetrievalConfig", "RetrievalResult", "RetrievalOrchestrator"):
        import src.retrieval.orchestrator as orch_mod
        return getattr(orch_mod, name)
    if name in ("DEFAULT_RAG_SYSTEM_PROMPT", "build_rag_user_prompt", "build_rag_prompt_messages"):
        import src.retrieval.prompts as prompt_mod
        return getattr(prompt_mod, name)
    if name in ("CitationAudit", "EvaluationResult", "RAGJudge", "audit_citations"):
        import src.retrieval.evaluator as eval_mod
        return getattr(eval_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


