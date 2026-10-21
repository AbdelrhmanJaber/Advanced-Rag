from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector

from src.config import DB_CONFIG

logger = logging.getLogger(__name__)


class PostgresVectorClient:
    """
    Client for managing documents, chunks, vectors, and metadata in PostgreSQL + pgvector.
    """

    def __init__(self, connection_string: Optional[str] = None) -> None:
        self.conn_str = connection_string or DB_CONFIG.connection_string

    def get_connection(self) -> psycopg.Connection:
        conn = psycopg.connect(self.conn_str)
        register_vector(conn)
        return conn

    def upsert_document(self, doc_id: str, doc_name: str, tree_path: str, total_chunks: int = 0) -> None:
        """Insert or update a document-level master record."""
        query = """
            INSERT INTO documents (doc_id, doc_name, tree_path, total_chunks)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                doc_name = EXCLUDED.doc_name,
                tree_path = EXCLUDED.tree_path,
                total_chunks = EXCLUDED.total_chunks;
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (doc_id, doc_name, tree_path, total_chunks))
            conn.commit()

    def upsert_chunks_batch(self, chunks: List[Dict[str, Any]], vectors: List[List[float]]) -> int:
        """
        Batch upsert chunks with their metadata, vectors, and tree_path pointers.
        """
        if not chunks:
            return 0

        query = """
            INSERT INTO document_chunks (
                chunk_id, doc_id, doc_name, chunk_index, text, embed_text,
                page, node_id, parent_id, parent_title, breadcrumb, level,
                node_type, char_count, token_count_est,
                has_table, has_image, table_markdown, table_html,
                image_path, caption, tree_path, embedding
            ) VALUES (
                %(chunk_id)s, %(doc_id)s, %(doc_name)s, %(chunk_index)s, %(text)s, %(embed_text)s,
                %(page)s, %(node_id)s, %(parent_id)s, %(parent_title)s, %(breadcrumb)s, %(level)s,
                %(node_type)s, %(char_count)s, %(token_count_est)s,
                %(has_table)s, %(has_image)s, %(table_markdown)s, %(table_html)s,
                %(image_path)s, %(caption)s, %(tree_path)s, %(embedding)s
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                text = EXCLUDED.text,
                embed_text = EXCLUDED.embed_text,
                page = EXCLUDED.page,
                parent_title = EXCLUDED.parent_title,
                breadcrumb = EXCLUDED.breadcrumb,
                has_table = EXCLUDED.has_table,
                has_image = EXCLUDED.has_image,
                table_markdown = EXCLUDED.table_markdown,
                table_html = EXCLUDED.table_html,
                image_path = EXCLUDED.image_path,
                caption = EXCLUDED.caption,
                tree_path = EXCLUDED.tree_path,
                embedding = EXCLUDED.embedding;
        """

        records = []
        for c, vec in zip(chunks, vectors):
            meta = c.get("metadata", {})
            chunk_id_str = str(c.get("chunk_id", ""))
            doc_id = (
                meta.get("doc_id")
                or meta.get("doc_name")
                or (chunk_id_str.split("_node_")[0] if "_node_" in chunk_id_str else chunk_id_str)
                or "unknown_doc"
            )
            records.append({
                "chunk_id": c["chunk_id"],
                "doc_id": doc_id,
                "doc_name": meta.get("doc_name", "unknown"),
                "chunk_index": c.get("chunk_index", 0),
                "text": c["text"],
                "embed_text": c.get("embed_text", c["text"]),
                "page": meta.get("page", 1),
                "node_id": meta.get("node_id", ""),
                "parent_id": meta.get("parent_id"),
                "parent_title": meta.get("parent_title"),
                "breadcrumb": meta.get("breadcrumb", ""),
                "level": meta.get("level", 4),
                "node_type": meta.get("node_type", "Paragraph"),
                "char_count": meta.get("char_count", len(c["text"])),
                "token_count_est": meta.get("token_count_est", 1),
                "has_table": meta.get("has_table", False),
                "has_image": meta.get("has_image", False),
                "table_markdown": meta.get("table_markdown"),
                "table_html": meta.get("table_html"),
                "image_path": meta.get("image_path"),
                "caption": meta.get("caption"),
                "tree_path": meta.get("tree_path", ""),
                "embedding": vec,
            })

        # PDF extraction can leave NUL (0x00) bytes in text; PostgreSQL TEXT rejects them.
        for rec in records:
            for key, value in rec.items():
                if isinstance(value, str) and "\x00" in value:
                    rec[key] = value.replace("\x00", "")

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, records)
            conn.commit()

        return len(records)

    def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 5,
        doc_name: Optional[str] = None,
        parent_title: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Cosine similarity search using pgvector's <=> cosine distance operator.
        Returns top matching chunks with metadata, tree_path, and similarity scores.
        """
        where_clauses = []
        where_params: List[Any] = []

        if doc_name:
            where_clauses.append(sql.SQL("doc_name = %s"))
            where_params.append(doc_name)
        if parent_title:
            where_clauses.append(sql.SQL("parent_title = %s"))
            where_params.append(parent_title)

        where_sql = (
            sql.SQL("WHERE {}").format(sql.SQL(" AND ").join(where_clauses))
            if where_clauses
            else sql.SQL("")
        )

        query = sql.SQL("""
            SELECT 
                chunk_id, doc_id, doc_name, chunk_index, text,
                page, node_id, parent_id, parent_title, breadcrumb,
                has_table, table_markdown, image_path, tree_path,
                1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            {}
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
        """).format(where_sql)
        all_params = [query_vector] + where_params + [query_vector, top_k]

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, all_params)
                rows = cur.fetchall()

        results = []
        for r in rows:
            results.append({
                "chunk_id": r[0],
                "doc_id": r[1],
                "doc_name": r[2],
                "chunk_index": r[3],
                "text": r[4],
                "page": r[5],
                "node_id": r[6],
                "parent_id": r[7],
                "parent_title": r[8],
                "breadcrumb": r[9],
                "has_table": r[10],
                "table_markdown": r[11],
                "image_path": r[12],
                "tree_path": r[13],
                "similarity": float(r[14]) if r[14] is not None else 0.0,
            })
        return results

    def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 10,
        candidate_pool: int = 50,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        doc_name: Optional[str] = None,
        parent_title: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval in a single round-trip:
          1. Dense:  top `candidate_pool` chunks by cosine distance (HNSW index, `<=>`).
          2. Sparse: top `candidate_pool` chunks by full-text rank (GIN index, `@@`).
          3. Fusion: Reciprocal Rank Fusion,
                score = dense_weight / (rrf_k + dense_rank) + sparse_weight / (rrf_k + sparse_rank)
             A chunk missing from one list simply gets 0 from that side.

        `query_text` is used for the keyword side only. Its terms are OR-ed together
        (stemmed, English stopwords removed) so long natural-language questions still match.
        `query_vector` must come from the same embedding model used at indexing time.
        """
        if top_k <= 0 or candidate_pool <= 0:
            return []
        candidate_pool = max(candidate_pool, top_k)

        filters: List[sql.Composable] = []
        params: Dict[str, Any] = {
            "qvec": query_vector,
            "qtext": query_text or "",
            "pool": candidate_pool,
            "top_k": top_k,
            "rrf_k": rrf_k,
            "dw": dense_weight,
            "sw": sparse_weight,
        }
        if doc_name:
            filters.append(sql.SQL("AND doc_name = %(doc_name)s"))
            params["doc_name"] = doc_name
        if parent_title:
            filters.append(sql.SQL("AND parent_title = %(parent_title)s"))
            params["parent_title"] = parent_title
        filter_sql = sql.SQL(" ").join(filters)

        # plainto_tsquery gives 'a' & 'b' & 'c' (AND). For questions that is too strict,
        # so we turn it into 'a' | 'b' | 'c' and let ts_rank_cd reward chunks matching more terms.
        tsq = sql.SQL(
            "replace(plainto_tsquery('english', %(qtext)s)::text, '&', '|')::tsquery"
        )

        query = sql.SQL("""
            WITH dense AS (
                SELECT chunk_id, distance,
                       ROW_NUMBER() OVER (ORDER BY distance, chunk_id) AS rnk
                FROM (
                    SELECT chunk_id, embedding <=> %(qvec)s::vector AS distance
                    FROM document_chunks
                    WHERE embedding IS NOT NULL {filters}
                    ORDER BY embedding <=> %(qvec)s::vector
                    LIMIT %(pool)s
                ) d
            ),
            sparse AS (
                SELECT chunk_id, kw_score,
                       ROW_NUMBER() OVER (ORDER BY kw_score DESC, chunk_id) AS rnk
                FROM (
                    SELECT chunk_id,
                           ts_rank_cd(text_search_tsv, {tsq}, 1) AS kw_score
                    FROM document_chunks
                    WHERE text_search_tsv @@ {tsq} {filters}
                    ORDER BY kw_score DESC
                    LIMIT %(pool)s
                ) s
            ),
            fused AS (
                SELECT
                    COALESCE(d.chunk_id, s.chunk_id) AS chunk_id,
                    d.rnk AS dense_rnk,
                    s.rnk AS sparse_rnk,
                    d.distance,
                    s.kw_score,
                    COALESCE(%(dw)s / (%(rrf_k)s + d.rnk), 0)
                  + COALESCE(%(sw)s / (%(rrf_k)s + s.rnk), 0) AS rrf_score
                FROM dense d
                FULL OUTER JOIN sparse s ON d.chunk_id = s.chunk_id
                ORDER BY rrf_score DESC, chunk_id
                LIMIT %(top_k)s
            )
            SELECT
                c.chunk_id, c.doc_id, c.doc_name, c.chunk_index, c.text,
                c.page, c.node_id, c.parent_id, c.parent_title, c.breadcrumb,
                c.node_type, c.has_table, c.table_markdown, c.has_image,
                c.image_path, c.caption, c.tree_path,
                f.rrf_score, f.dense_rnk, f.sparse_rnk, f.distance, f.kw_score
            FROM fused f
            JOIN document_chunks c ON c.chunk_id = f.chunk_id
            ORDER BY f.rrf_score DESC, c.chunk_id;
        """).format(filters=filter_sql, tsq=tsq)

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # HNSW only explores `ef_search` candidates (default 40); it must be >= the pool
                # or the dense side silently returns fewer rows. SET LOCAL scope = this transaction.
                cur.execute(
                    "SELECT set_config('hnsw.ef_search', %s, true)",
                    (str(min(max(40, candidate_pool), 1000)),),
                )
                if filters:
                    # With WHERE filters, HNSW may find too few matching rows in one pass.
                    # Iterative scan (pgvector >= 0.8) keeps searching until LIMIT is satisfied.
                    cur.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
                # On small/medium tables (thousands of rows) pgvector's cost model often prefers
                # a full sequential scan + sort, even though HNSW is much faster. Steer the planner
                # to the indexes for this transaction only.
                cur.execute("SELECT set_config('enable_seqscan', 'off', true)")
                cur.execute(query, params)
                rows = cur.fetchall()

        results: List[Dict[str, Any]] = []
        for r in rows:
            dense_rnk, sparse_rnk, distance, kw_score = r[18], r[19], r[20], r[21]
            if dense_rnk is not None and sparse_rnk is not None:
                source = "both"
            elif dense_rnk is not None:
                source = "dense"
            else:
                source = "sparse"
            results.append({
                "chunk_id": r[0],
                "doc_id": r[1],
                "doc_name": r[2],
                "chunk_index": r[3],
                "text": r[4],
                "page": r[5],
                "node_id": r[6],
                "parent_id": r[7],
                "parent_title": r[8],
                "breadcrumb": r[9],
                "node_type": r[10],
                "has_table": r[11],
                "table_markdown": r[12],
                "has_image": r[13],
                "image_path": r[14],
                "caption": r[15],
                "tree_path": r[16],
                "rrf_score": float(r[17]),
                "dense_rank": dense_rnk,
                "sparse_rank": sparse_rnk,
                "similarity": (1.0 - float(distance)) if distance is not None else None,
                "keyword_score": float(kw_score) if kw_score is not None else None,
                "source": source,
            })
        return results

    def get_surrounding_chunks(
        self,
        doc_name: str,
        chunk_index: int,
        window: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Fetch surrounding chunks (window before and window after) for context expansion.
        """
        if window <= 0:
            return []
        min_idx = max(0, chunk_index - window)
        max_idx = chunk_index + window

        query = """
            SELECT
                chunk_id, doc_id, doc_name, chunk_index, text,
                page, node_id, parent_id, parent_title, breadcrumb,
                node_type, has_table, table_markdown, has_image,
                image_path, caption, tree_path
            FROM document_chunks
            WHERE doc_name = %s AND chunk_index >= %s AND chunk_index <= %s
            ORDER BY chunk_index ASC;
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (doc_name, min_idx, max_idx))
                rows = cur.fetchall()

        return [
            {
                "chunk_id": r[0],
                "doc_id": r[1],
                "doc_name": r[2],
                "chunk_index": r[3],
                "text": r[4],
                "page": r[5],
                "node_id": r[6],
                "parent_id": r[7],
                "parent_title": r[8],
                "breadcrumb": r[9],
                "node_type": r[10],
                "has_table": r[11],
                "table_markdown": r[12],
                "has_image": r[13],
                "image_path": r[14],
                "caption": r[15],
                "tree_path": r[16],
            }
            for r in rows
        ]

