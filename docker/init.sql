-- PostgreSQL pgvector initialization schema
CREATE EXTENSION IF NOT EXISTS vector;

-- Table for storing document-level master records
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    doc_name TEXT NOT NULL,
    tree_path TEXT NOT NULL,
    total_chunks INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing individual chunks with vectors and metadata
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
    doc_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embed_text TEXT NOT NULL,
    
    -- Context & Lineage Metadata
    page INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    parent_id TEXT,
    parent_title TEXT,
    breadcrumb TEXT NOT NULL,
    level INTEGER NOT NULL,
    node_type TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    token_count_est INTEGER NOT NULL,
    
    -- Structured Content Payloads
    has_table BOOLEAN DEFAULT FALSE,
    has_image BOOLEAN DEFAULT FALSE,
    table_markdown TEXT,
    table_html TEXT,
    image_path TEXT,
    caption TEXT,
    tree_path TEXT NOT NULL,

    -- Vector column. HNSW indexes require a FIXED dimension.
    -- 768 = nomic-embed-text (see src/indexing/embedder.py). Change this if you switch models.
    embedding vector(768),

    -- Sparse / keyword search column (auto-maintained by Postgres on INSERT/UPDATE).
    -- Weights: A = section titles, B = body + tables, C = captions.
    text_search_tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(breadcrumb, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(parent_title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(text, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(table_markdown, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(caption, '')), 'C')
    ) STORED,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for high-traffic metadata filtering
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_name ON document_chunks(doc_name);
CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON document_chunks(parent_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON document_chunks(page);
CREATE INDEX IF NOT EXISTS idx_chunks_node_type ON document_chunks(node_type);
CREATE INDEX IF NOT EXISTS idx_chunks_has_table ON document_chunks(has_table);

-- Hybrid retrieval indexes
-- Dense: HNSW graph index for cosine distance (<=>)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Sparse: GIN inverted index for full-text keyword search (@@)
CREATE INDEX IF NOT EXISTS idx_chunks_tsv
    ON document_chunks USING GIN (text_search_tsv);
