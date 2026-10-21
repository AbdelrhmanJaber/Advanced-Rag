-- Migration 001: prepare document_chunks for hybrid (dense + sparse) retrieval.
-- Safe to re-run. Mirrors the schema in docker/init.sql for databases created before it changed.
-- Apply with:
--   docker exec -i rag_pgvector_db psql -U rag_user -d rag_db -v ON_ERROR_STOP=1 < docker/migrations/001_hybrid_indexes.sql

BEGIN;

-- 1. Fix the embedding dimension (HNSW cannot index an unsized `vector`).
--    Fails loudly if any existing row is not 768-dim, which is what we want.
ALTER TABLE document_chunks
    ALTER COLUMN embedding TYPE vector(768);

-- 2. Weighted full-text column, maintained automatically by Postgres.
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS text_search_tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(breadcrumb, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(parent_title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(text, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(table_markdown, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(caption, '')), 'C')
    ) STORED;

-- 3. Dense index (cosine).
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 4. Sparse index.
CREATE INDEX IF NOT EXISTS idx_chunks_tsv
    ON document_chunks USING GIN (text_search_tsv);

COMMIT;
