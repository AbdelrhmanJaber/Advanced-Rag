from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.indexing.embedder import OllamaEmbedder
from src.indexing.db_client import PostgresVectorClient

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Orchestrates reading chunk JSONs, computing embeddings with nomic-embed-text,
    attaching the tree.json path, and batch upserting into PostgreSQL + pgvector.
    """

    def __init__(
        self,
        embedder: Optional[OllamaEmbedder] = None,
        db_client: Optional[PostgresVectorClient] = None,
        batch_size: int = 32,
    ) -> None:
        self.embedder = embedder or OllamaEmbedder()
        self.db_client = db_client or PostgresVectorClient()
        self.batch_size = batch_size

    def index_document_files(
        self,
        chunks_json_path: str | Path,
        tree_json_path: Optional[str | Path] = None,
    ) -> int:
        chunks_file = Path(chunks_json_path)
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

        # Derive tree_path if not provided
        if tree_json_path is None:
            # e.g., doc_0_chunks.json -> doc_0_tree.json
            stem_prefix = chunks_file.name.replace("_chunks.json", "")
            tree_file = chunks_file.parent / f"{stem_prefix}_tree.json"
            tree_path_str = str(tree_file.resolve()) if tree_file.exists() else str(chunks_file.parent)
        else:
            tree_path_str = str(Path(tree_json_path).resolve())

        # Load chunks
        chunks: List[Dict[str, Any]] = json.loads(chunks_file.read_text(encoding="utf-8"))
        if not chunks:
            logger.warning("No chunks found in %s", chunks_file)
            return 0

        # Inject tree_path into each chunk's metadata
        for c in chunks:
            if "metadata" not in c:
                c["metadata"] = {}
            c["metadata"]["tree_path"] = tree_path_str

        # Register document master record
        doc_name = chunks[0]["metadata"].get("doc_name", chunks_file.stem)
        doc_id = chunks[0]["metadata"].get("doc_id", doc_name)
        self.db_client.upsert_document(
            doc_id=doc_id,
            doc_name=doc_name,
            tree_path=tree_path_str,
            total_chunks=len(chunks),
        )

        # Batch embed and upsert
        total_indexed = 0
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts_to_embed = [c.get("embed_text", c["text"]) for c in batch]
            vectors = self.embedder.embed_batch(texts_to_embed)
            indexed_count = self.db_client.upsert_chunks_batch(batch, vectors)
            total_indexed += indexed_count
            logger.info("Indexed %d/%d chunks for %s", total_indexed, len(chunks), doc_name)

        return total_indexed

    def index_output_directory(self, output_dir: str | Path) -> Dict[str, int]:
        """
        Scan output directory for all *_chunks.json and index them into pgvector.
        """
        out_path = Path(output_dir)
        results: Dict[str, int] = {}

        chunk_files = list(out_path.rglob("*_chunks.json"))
        logger.info("Found %d chunk files in %s", len(chunk_files), output_dir)

        for cf in chunk_files:
            try:
                count = self.index_document_files(cf)
                results[cf.name] = count
            except Exception as e:
                logger.exception("Failed to index %s: %s", cf, e)
                results[cf.name] = -1

        return results
