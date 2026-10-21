from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence
import ollama
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# Official prompt prefix recommended by Mixedbread AI for mxbai-embed-large queries
MXBAI_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _dot_product(v1: Sequence[float], v2: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(a * a for a in v))


def _cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    n1 = _norm(v1)
    n2 = _norm(v2)
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 0.0
    cos = _dot_product(v1, v2) / (n1 * n2)
    # Clamp to [-1.0, 1.0] to guard against floating-point imprecision
    return max(-1.0, min(1.0, cos))


class MxbaiReranker:
    """
    Second-stage reranker powered by Ollama's `mxbai-embed-large:latest` (1024-dimension SOTA model).
    
    Workflow:
      1. Formats the query with Mixedbread's query representation prompt.
      2. Batches the candidate chunks and computes their 1024-dim embeddings via Ollama.
      3. Computes normalized cosine similarities between the query and each candidate.
      4. Reranks the candidates from highest to lowest semantic relevance.
    """

    def __init__(self, model_name: str = "mxbai-embed-large:latest") -> None:
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        text_field: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Rerank a pool of candidate chunks.
        
        Args:
            query: The user's natural language question.
            candidates: List of candidate chunk dictionaries (from Stage 1 hybrid retrieval).
            top_k: Number of final reranked chunks to return.
            text_field: Key in the candidate dict to score ("text" or "embed_text").
            
        Returns:
            The top_k candidates sorted by `rerank_score` in descending order.
        """
        # Edge Case 1: Empty candidate list or non-positive top_k
        if not candidates or top_k <= 0:
            return []

        # Edge Case 2: Empty or whitespace query
        if not query or not query.strip():
            logger.warning("Empty query passed to reranker; returning candidates without re-scoring.")
            fallback: List[Dict[str, Any]] = []
            for c in candidates[:top_k]:
                entry = dict(c)
                entry.setdefault("rerank_score", 0.0)
                entry.setdefault("rerank_model", self.model_name)
                fallback.append(entry)
            return fallback

        # 1. Embed query with official Mixedbread retrieval instruction
        formatted_query = f"{MXBAI_QUERY_PREFIX}{query.strip()}"
        query_resp = ollama.embed(model=self.model_name, input=formatted_query)
        if not query_resp or not query_resp.embeddings:
            logger.warning("Ollama returned empty query embedding; returning original candidates.")
            return candidates[:top_k]
        query_vec: Sequence[float] = query_resp.embeddings[0]

        # 2. Extract passage texts from candidate chunks
        passages: List[str] = []
        for c in candidates:
            # Use breadcrumb + text if available for enhanced contextual reranking
            val = c.get(text_field) or c.get("text") or ""
            txt = str(val).strip()
            # Edge Case 3: Empty passage text
            if not txt:
                txt = "empty"

            breadcrumb = c.get("breadcrumb")
            if breadcrumb and not txt.startswith("["):
                full_passage = f"[{breadcrumb}]\n{txt}"
            else:
                full_passage = txt
            passages.append(full_passage)

        # 3. Batch embed all passages in a single API round-trip
        passage_resp = ollama.embed(model=self.model_name, input=passages)
        if not passage_resp or not passage_resp.embeddings:
            logger.warning("Ollama returned empty passage embeddings; returning original candidates.")
            return candidates[:top_k]
        passage_vecs: Sequence[Sequence[float]] = passage_resp.embeddings

        # 4. Score each candidate (safe pairing against length mismatches)
        scored_candidates: List[Dict[str, Any]] = []
        for i, c in enumerate(candidates):
            if i < len(passage_vecs):
                p_vec: Sequence[float] = passage_vecs[i]
                score = _cosine_similarity(query_vec, p_vec)
            else:
                score = 0.0

            entry = dict(c)
            entry["rerank_score"] = float(score)
            entry["rerank_model"] = self.model_name
            scored_candidates.append(entry)

        # 5. Sort descending by rerank score
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored_candidates[:top_k]


class CrossEncoderReranker:
    """
    True Cross-Encoder reranker using HuggingFace Transformers.
    
    Computes full token-level cross-attention across [CLS] Query [SEP] Passage [SEP] pairs
    to output precise relevance scores.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def model(self) -> Any:
        if self._model is None:
            loaded_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            loaded_model.to(self.device)
            loaded_model.eval()
            self._model = loaded_model
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        text_field: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using token-level cross-attention.
        """
        if not candidates or top_k <= 0:
            return []

        if not query or not query.strip():
            logger.warning("Empty query passed to cross-encoder; returning candidates without re-scoring.")
            fallback: List[Dict[str, Any]] = []
            for c in candidates[:top_k]:
                entry = dict(c)
                entry.setdefault("rerank_score", 0.0)
                entry.setdefault("rerank_model", self.model_name)
                fallback.append(entry)
            return fallback

        # Extract passages with context breadcrumbs
        passages: List[str] = []
        for c in candidates:
            val = c.get(text_field) or c.get("text") or ""
            txt = str(val).strip()
            if not txt:
                txt = "empty"

            breadcrumb = c.get("breadcrumb")
            if breadcrumb and not txt.startswith("["):
                full_passage = f"[{breadcrumb}]\n{txt}"
            else:
                full_passage = txt
            passages.append(full_passage)

        clean_query = query.strip()
        pairs = [[clean_query, p] for p in passages]

        all_scores: List[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                if logits.shape[-1] == 1:
                    raw_scores = logits.view(-1).float()
                    batch_scores = torch.sigmoid(raw_scores).cpu().tolist()
                else:
                    batch_scores = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()

            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            all_scores.extend(batch_scores)

        scored_candidates: List[Dict[str, Any]] = []
        for i, c in enumerate(candidates):
            entry = dict(c)
            entry["rerank_score"] = float(all_scores[i]) if i < len(all_scores) else 0.0
            entry["rerank_model"] = self.model_name
            scored_candidates.append(entry)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_k]

