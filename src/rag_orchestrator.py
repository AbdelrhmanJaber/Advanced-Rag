from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from multiprocessing.synchronize import Lock as LockType
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.indexing.indexer import DocumentIndexer
from src.ingestion.loader import LocalFileLoader
from src.ingestion.validator import validator
from src.preprocessing.classifier import classifier
from src.preprocessing.native_parser import NativeDocumentParser
from src.preprocessing.scanned_parser import ScannedDocumentParser
from src.retrieval.evaluator import EvaluationResult
from src.retrieval.orchestrator import (
    RetrievalConfig,
    RetrievalOrchestrator,
    RetrievalResult,
)

logger = logging.getLogger("rag_orchestrator")
SENTINEL = None


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass
class RAGPipelineConfig:
    """Master configuration controlling both offline parallel ingestion and online retrieval."""
    # Offline Ingestion & Parallelism Settings
    raw_dir: str = "data"
    output_dir: str = "output"
    n_validator_workers: int = 4
    n_classifier_workers: int = 4
    n_native_parser_workers: int = 4
    n_scanned_parser_workers: int = 2
    n_indexer_workers: int = 2
    queue_max_size: int = 200
    embed_batch_size: int = 32

    # Online Retrieval & Generation Settings
    top_k: int = 5
    candidate_pool: int = 30
    expand_window: int = 0
    reorder_lost_in_middle: bool = True
    synthesis_model: str = "llama3.2:latest"
    judge_model: str = "qwen2.5:latest"
    evaluate: bool = True


# =============================================================================
# Worker Stage Implementations with Pipelined Parallelism
# =============================================================================

def lock_and_release(
    counter: Any,
    lock: LockType,
    downstream: List[Tuple[mp.Queue, int]],
) -> None:
    """
    Safely decrements the worker counter under lock and injects SENTINEL signals
    into downstream queues once all upstream workers of this stage complete.
    """
    with lock:
        counter.value -= 1
        if counter.value == 0:
            for q, n_consumers in downstream:
                for _ in range(n_consumers):
                    q.put(SENTINEL)


def loader_stage(raw_dir: str, q_out: mp.Queue, n_validator_workers: int) -> None:
    """Stage 1: Discovers raw PDF documents and pushes paths to the validator queue."""
    loader = LocalFileLoader(raw_dir)
    count = 0
    try:
        for file_path in loader.scan_files():
            q_out.put(str(file_path))
            count += 1
    finally:
        logger.info("[loader] Discovered %d files to ingest.", count)
        for _ in range(n_validator_workers):
            q_out.put(SENTINEL)


def validator_stage(
    worker_id: int,
    q_in: mp.Queue,
    q_out: mp.Queue,
    counter: Any,
    lock: LockType,
    downstream: List[Tuple[mp.Queue, int]],
) -> None:
    """Stage 2: Validates PDF headers, checks for corruption, and filters corrupted files."""
    ok, rejected, errors = 0, 0, 0
    while True:
        item = q_in.get()
        if item == SENTINEL:
            break
        try:
            is_valid, _metadata, msg = validator(item)
            if is_valid:
                q_out.put(item)
                ok += 1
            else:
                logger.warning("[validator-%d] Rejected corrupt file %s: %s", worker_id, item, msg)
                rejected += 1
        except Exception:
            logger.exception("[validator-%d] Error on file %s", worker_id, item)
            errors += 1

    logger.info("[validator-%d] Done: ok=%d rejected=%d errors=%d", worker_id, ok, rejected, errors)
    lock_and_release(counter, lock, downstream)


def classifier_stage(
    worker_id: int,
    q_in: mp.Queue,
    q_native: mp.Queue,
    q_scanned: mp.Queue,
    counter: Any,
    lock: LockType,
    downstream: List[Tuple[mp.Queue, int]],
) -> None:
    """Stage 3: Classifies valid PDFs into NATIVE (digital) vs. SCANNED (OCR needed)."""
    native_count, scanned_count, errors = 0, 0, 0
    while True:
        item = q_in.get()
        if item == SENTINEL:
            break
        try:
            meta = classifier(item)
            if meta.get("file_type") == "NATIVE":
                q_native.put(item)
                native_count += 1
            else:
                q_scanned.put(item)
                scanned_count += 1
        except Exception:
            logger.exception("[classifier-%d] Error on file %s", worker_id, item)
            errors += 1

    logger.info("[classifier-%d] Done: native=%d scanned=%d errors=%d", worker_id, native_count, scanned_count, errors)
    lock_and_release(counter, lock, downstream)


def native_parse_stage(
    worker_id: int,
    q_in: mp.Queue,
    q_indexer: mp.Queue,
    output_dir: str,
    counter: Any,
    lock: LockType,
    downstream: List[Tuple[mp.Queue, int]],
) -> None:
    """Stage 4a: Fast PyMuPDF font/layout parsing, document tree building, and semantic chunking."""
    ok, errors = 0, 0
    while True:
        item = q_in.get()
        if item == SENTINEL:
            break
        try:
            file_output_dir = Path(output_dir) / "native" / Path(item).stem
            parser = NativeDocumentParser(item, output_dir=str(file_output_dir))
            tree_path = parser.parse_and_save_json()
            chunks_path = Path(tree_path).parent / f"{Path(item).stem}_chunks.json"

            # Stream directly to indexer queue for real-time pipelining
            q_indexer.put((str(chunks_path.resolve()), tree_path))
            ok += 1
        except Exception:
            logger.exception("[native-%d] Error parsing %s", worker_id, item)
            errors += 1

    logger.info("[native-%d] Done: parsed=%d errors=%d", worker_id, ok, errors)
    lock_and_release(counter, lock, downstream)


def scanned_parse_stage(
    worker_id: int,
    q_in: mp.Queue,
    q_indexer: mp.Queue,
    output_dir: str,
    counter: Any,
    lock: LockType,
    downstream: List[Tuple[mp.Queue, int]],
) -> None:
    """Stage 4b: OCR text extraction, layout analysis, document tree, and semantic chunking."""
    ok, errors = 0, 0
    while True:
        item = q_in.get()
        if item == SENTINEL:
            break
        try:
            file_output_dir = Path(output_dir) / "scanned" / Path(item).stem
            parser = ScannedDocumentParser(item, output_dir=str(file_output_dir))
            tree_path = parser.parse_and_save_json()
            chunks_path = Path(tree_path).parent / f"{Path(item).stem}_chunks.json"

            # Stream directly to indexer queue
            q_indexer.put((str(chunks_path.resolve()), tree_path))
            ok += 1
        except Exception:
            logger.exception("[scanned-%d] Error OCR parsing %s", worker_id, item)
            errors += 1

    logger.info("[scanned-%d] Done: ocr_parsed=%d errors=%d", worker_id, ok, errors)
    lock_and_release(counter, lock, downstream)


def indexer_stage(
    worker_id: int,
    q_in: mp.Queue,
    batch_size: int,
) -> None:
    """
    Stage 5: Parallel embedding (nomic-embed-text) & batch PostgreSQL/pgvector upsert.
    Consumes chunk files as soon as parsers produce them, achieving continuous streaming parallelism.
    """
    indexer = DocumentIndexer(batch_size=batch_size)
    total_docs = 0
    total_chunks = 0

    while True:
        item = q_in.get()
        if item == SENTINEL:
            break
        try:
            chunks_path, tree_path = item
            count = indexer.index_document_files(chunks_path, tree_path)
            total_docs += 1
            total_chunks += count
            logger.info("[indexer-%d] Indexed doc %s (%d chunks)", worker_id, Path(chunks_path).stem, count)
        except Exception:
            logger.exception("[indexer-%d] Failed indexing chunk item: %s", worker_id, item)

    logger.info("[indexer-%d] Done: total_docs=%d total_chunks=%d", worker_id, total_docs, total_chunks)


# =============================================================================
# Master End-to-End Orchestrator
# =============================================================================

class MasterRAGOrchestrator:
    """
    Production Master Orchestrator:
      - Coordinates the complete offline parallel ingestion and indexing pipeline.
      - Executes the complete online two-stage hybrid retrieval, reranking, synthesis, and evaluation.
    """

    def __init__(self, config: Optional[RAGPipelineConfig] = None) -> None:
        self.config = config or RAGPipelineConfig()
        self.retrieval_orchestrator = RetrievalOrchestrator()

    def ingest_and_index(
        self,
        raw_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes the 5-stage parallel ingestion, classification, parsing, and indexing pipeline.
        Utilizes streaming multiprocessing queues so indexers process chunks in parallel with parsers.
        """
        cfg = self.config
        raw_path = raw_dir or cfg.raw_dir
        out_path = output_dir or cfg.output_dir

        t_start = time.perf_counter()
        logger.info(
            "Starting parallel ingestion pipeline: raw_dir='%s' output_dir='%s' | "
            "workers: val=%d class=%d nat=%d scan=%d idx=%d",
            raw_path, out_path,
            cfg.n_validator_workers, cfg.n_classifier_workers,
            cfg.n_native_parser_workers, cfg.n_scanned_parser_workers,
            cfg.n_indexer_workers,
        )

        # 1. Pipeline Queues
        q_validator: mp.Queue = mp.Queue(cfg.queue_max_size)
        q_classifier: mp.Queue = mp.Queue(cfg.queue_max_size)
        q_native: mp.Queue = mp.Queue(cfg.queue_max_size)
        q_scanned: mp.Queue = mp.Queue(cfg.queue_max_size)
        q_indexer: mp.Queue = mp.Queue(cfg.queue_max_size)

        # 2. Stage Synchronization Counters & Locks
        val_counter = mp.Value("i", cfg.n_validator_workers)
        val_lock = mp.Lock()

        class_counter = mp.Value("i", cfg.n_classifier_workers)
        class_lock = mp.Lock()

        # Shared counter for both native and scanned parsers to safely trigger indexer sentinels
        total_parsers = cfg.n_native_parser_workers + cfg.n_scanned_parser_workers
        parser_counter = mp.Value("i", total_parsers)
        parser_lock = mp.Lock()

        processes: List[mp.Process] = []

        # Stage 1: Loader
        processes.append(
            mp.Process(
                target=loader_stage,
                args=(raw_path, q_validator, cfg.n_validator_workers),
                name="loader",
            )
        )

        # Stage 2: Validators
        for i in range(cfg.n_validator_workers):
            processes.append(
                mp.Process(
                    target=validator_stage,
                    args=(
                        i, q_validator, q_classifier, val_counter, val_lock,
                        [(q_classifier, cfg.n_classifier_workers)],
                    ),
                    name=f"validator-{i}",
                )
            )

        # Stage 3: Classifiers
        for i in range(cfg.n_classifier_workers):
            processes.append(
                mp.Process(
                    target=classifier_stage,
                    args=(
                        i, q_classifier, q_native, q_scanned, class_counter, class_lock,
                        [(q_native, cfg.n_native_parser_workers), (q_scanned, cfg.n_scanned_parser_workers)],
                    ),
                    name=f"classifier-{i}",
                )
            )

        # Stage 4a: Native Parsers
        for i in range(cfg.n_native_parser_workers):
            processes.append(
                mp.Process(
                    target=native_parse_stage,
                    args=(
                        i, q_native, q_indexer, out_path, parser_counter, parser_lock,
                        [(q_indexer, cfg.n_indexer_workers)],
                    ),
                    name=f"native-parser-{i}",
                )
            )

        # Stage 4b: Scanned OCR Parsers
        for i in range(cfg.n_scanned_parser_workers):
            processes.append(
                mp.Process(
                    target=scanned_parse_stage,
                    args=(
                        i, q_scanned, q_indexer, out_path, parser_counter, parser_lock,
                        [(q_indexer, cfg.n_indexer_workers)],
                    ),
                    name=f"scanned-parser-{i}",
                )
            )

        # Stage 5: Parallel Indexers (Streaming embeddings & pgvector upserts)
        for i in range(cfg.n_indexer_workers):
            processes.append(
                mp.Process(
                    target=indexer_stage,
                    args=(i, q_indexer, cfg.embed_batch_size),
                    name=f"indexer-{i}",
                )
            )

        # Launch all workers concurrently
        for p in processes:
            p.start()

        # Await completion
        for p in processes:
            p.join()

        elapsed = time.perf_counter() - t_start
        logger.info("Parallel ingestion & indexing pipeline completed in %.2fs", elapsed)

        return {
            "status": "success",
            "elapsed_seconds": round(elapsed, 2),
            "raw_dir": raw_path,
            "output_dir": out_path,
        }

    def ask(
        self,
        query: str,
        evaluate: Optional[bool] = None,
        synthesis_model: Optional[str] = None,
        judge_model: Optional[str] = None,
        top_k: Optional[int] = None,
        expand_window: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end query workflow:
          1. Dual Dense (HNSW) + Sparse (GIN) hybrid search with RRF
          2. Cross-Encoder token-level reranking
          3. Attention optimization via Lost-in-the-Middle reordering
          4. Grounded answer generation with strict inline citations & conflict rules
          5. Anti-biased LLM-as-a-Judge evaluation with deterministic citation audit
        """
        cfg = self.config
        retrieval_cfg = RetrievalConfig(
            top_k=top_k if top_k is not None else cfg.top_k,
            candidate_pool=cfg.candidate_pool,
            expand_window=expand_window if expand_window is not None else cfg.expand_window,
            reorder_lost_in_middle=cfg.reorder_lost_in_middle,
            evaluate=evaluate if evaluate is not None else cfg.evaluate,
            judge_model=judge_model or cfg.judge_model,
        )

        return self.retrieval_orchestrator.answer(
            query=query,
            config=retrieval_cfg,
            model=synthesis_model or cfg.synthesis_model,
            evaluate=evaluate if evaluate is not None else cfg.evaluate,
            judge_model=judge_model or cfg.judge_model,
        )


# =============================================================================
# CLI Interface
# =============================================================================

def _main() -> None:
    parser = argparse.ArgumentParser(description="End-to-End Master RAG Orchestrator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_p = subparsers.add_parser("ingest", help="Run parallel ingestion, parsing, and pgvector indexing")
    ingest_p.add_argument("--raw-dir", default="data", help="Directory of raw PDF files")
    ingest_p.add_argument("--output-dir", default="output", help="Output directory for trees and chunks")
    ingest_p.add_argument("--n-validators", type=int, default=4)
    ingest_p.add_argument("--n-classifiers", type=int, default=4)
    ingest_p.add_argument("--n-native", type=int, default=4)
    ingest_p.add_argument("--n-scanned", type=int, default=2)
    ingest_p.add_argument("--n-indexers", type=int, default=2)

    # Ask command
    ask_p = subparsers.add_parser("ask", help="Query the RAG pipeline with grounded generation & evaluation")
    ask_p.add_argument("query", help="User question or research query")
    ask_p.add_argument("--top-k", type=int, default=5)
    ask_p.add_argument("--model", default="llama3.2:latest", help="Synthesis model")
    ask_p.add_argument("--judge-model", default="qwen2.5:latest", help="Judge model (disjoint to avoid bias)")
    ask_p.add_argument("--no-eval", action="store_true", help="Disable LLM-as-a-Judge evaluation")

    # Run-all command (Ingest + Ask)
    run_all_p = subparsers.add_parser("run-all", help="Ingest documents and immediately answer a question")
    run_all_p.add_argument("--raw-dir", default="data")
    run_all_p.add_argument("--output-dir", default="output")
    run_all_p.add_argument("query", help="Question to ask after indexing")
    run_all_p.add_argument("--model", default="llama3.2:latest")
    run_all_p.add_argument("--judge-model", default="qwen2.5:latest")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(processName)s) %(message)s",
    )

    if args.command == "ingest":
        cfg = RAGPipelineConfig(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            n_validator_workers=args.n_validators,
            n_classifier_workers=args.n_classifiers,
            n_native_parser_workers=args.n_native,
            n_scanned_parser_workers=args.n_scanned,
            n_indexer_workers=args.n_indexers,
        )
        orch = MasterRAGOrchestrator(cfg)
        orch.ingest_and_index()

    elif args.command == "ask":
        orch = MasterRAGOrchestrator()
        res = orch.ask(
            query=args.query,
            evaluate=not args.no_eval,
            synthesis_model=args.model,
            judge_model=args.judge_model,
            top_k=args.top_k,
        )
        print("\n==================== LLM ANSWER ====================")
        print(res["answer"])
        print("\n==================== RETRIEVAL METRICS ====================")
        ret_obj: RetrievalResult = res["retrieval"]
        print(ret_obj.metrics)

        if "evaluation" in res:
            ev: EvaluationResult = res["evaluation"]
            print("\n==================== LLM-AS-A-JUDGE EVALUATION ====================")
            print(f"Passed: {ev.passed} | Overall Score: {ev.overall_score} (Judge: {ev.judge_model})")
            print(f"Faithfulness: {ev.faithfulness_score} - {ev.faithfulness_reasoning}")
            print(f"Relevance: {ev.relevance_score} - {ev.relevance_reasoning}")
            print(f"Conflict Handling: {ev.conflict_handling_score} - {ev.conflict_handling_reasoning}")
            print(f"Citation Audit: {ev.citation_audit.valid_citations}/{ev.citation_audit.total_citations} valid citations")
            if ev.same_model_bias_warning:
                print("WARNING: Same-model bias warning triggered!")

    elif args.command == "run-all":
        cfg = RAGPipelineConfig(raw_dir=args.raw_dir, output_dir=args.output_dir)
        orch = MasterRAGOrchestrator(cfg)
        orch.ingest_and_index()
        res = orch.ask(
            query=args.query,
            evaluate=True,
            synthesis_model=args.model,
            judge_model=args.judge_model,
        )
        print("\n==================== LLM ANSWER ====================")
        print(res["answer"])
        if "evaluation" in res:
            ev = res["evaluation"]
            print(f"\n[Judge: {ev.judge_model}] Passed: {ev.passed} | Overall Score: {ev.overall_score}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    _main()
