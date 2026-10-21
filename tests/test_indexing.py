import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import tempfile

from src.indexing.embedder import OllamaEmbedder
from src.indexing.db_client import PostgresVectorClient
from src.indexing.indexer import DocumentIndexer


class TestOllamaEmbedder(unittest.TestCase):

    @patch("src.indexing.embedder.ollama.embeddings")
    def test_embed_text_mocked(self, mock_embeddings):
        mock_embeddings.return_value = {"embedding": [0.1] * 768}
        embedder = OllamaEmbedder(model_name="nomic-embed-text")
        vec = embedder.embed_text("test sentence")

        self.assertEqual(len(vec), 768)
        self.assertEqual(vec[0], 0.1)
        mock_embeddings.assert_called_once_with(model="nomic-embed-text", prompt="test sentence")

    @patch("src.indexing.embedder.ollama.embeddings")
    def test_embed_batch_mocked(self, mock_embeddings):
        mock_embeddings.return_value = {"embedding": [0.2] * 768}
        embedder = OllamaEmbedder(model_name="nomic-embed-text")
        vectors = embedder.embed_batch(["text1", "text2"])

        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 768)
        self.assertEqual(mock_embeddings.call_count, 2)


class TestPostgresVectorClientAndIndexer(unittest.TestCase):

    def setUp(self):
        self.db_client = PostgresVectorClient()
        self.mock_embedder = MagicMock(spec=OllamaEmbedder)
        # Mock embedder returning 768-dim dummy vectors
        self.mock_embedder.embed_batch.side_effect = lambda texts: [[0.05 * (i + 1)] * 768 for i in range(len(texts))]
        self.mock_embedder.embed_text.return_value = [0.05] * 768
        with self.db_client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM documents WHERE doc_id IN ('test_doc_unit', 'sample.pdf', 'live_test_doc') OR doc_id LIKE 'test_%' OR doc_id LIKE 'live_%';"
                )
            conn.commit()

    def tearDown(self):
        with self.db_client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM documents WHERE doc_id IN ('test_doc_unit', 'sample.pdf', 'live_test_doc') OR doc_id LIKE 'test_%' OR doc_id LIKE 'live_%';"
                )
            conn.commit()

    def test_database_connection(self):
        conn = self.db_client.get_connection()
        self.assertIsNotNone(conn)
        conn.close()

    def test_document_and_chunks_upsert_and_similarity_search(self):
        doc_id = "test_doc_unit"
        doc_name = "test_doc_unit.pdf"
        tree_path = "/path/to/test_doc_unit_tree.json"

        # 1. Upsert document
        self.db_client.upsert_document(
            doc_id=doc_id,
            doc_name=doc_name,
            tree_path=tree_path,
            total_chunks=2,
        )

        # 2. Upsert chunks
        sample_chunks = [
            {
                "chunk_id": f"{doc_id}_chunk_0",
                "chunk_index": 0,
                "text": "Deep learning architectures for natural language processing.",
                "embed_text": f"[{doc_name} > NLP]\nDeep learning architectures for natural language processing.",
                "metadata": {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page": 1,
                    "node_id": "n_1",
                    "parent_id": "n_0",
                    "parent_title": "NLP",
                    "breadcrumb": f"{doc_name} > NLP",
                    "level": 4,
                    "node_type": "Paragraph",
                    "char_count": 60,
                    "token_count_est": 10,
                    "has_table": False,
                    "has_image": False,
                    "tree_path": tree_path,
                },
            },
            {
                "chunk_id": f"{doc_id}_chunk_1",
                "chunk_index": 1,
                "text": "Table 1 compares Transformer latency across different batch sizes.",
                "embed_text": f"[{doc_name} > Benchmarks]\nTable 1 compares Transformer latency across different batch sizes.",
                "metadata": {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page": 2,
                    "node_id": "n_2",
                    "parent_id": "n_0",
                    "parent_title": "Benchmarks",
                    "breadcrumb": f"{doc_name} > Benchmarks",
                    "level": 4,
                    "node_type": "Table",
                    "char_count": 67,
                    "token_count_est": 12,
                    "has_table": True,
                    "has_image": False,
                    "table_markdown": "| Model | Latency |\n|---|---|\n| BERT | 15ms |",
                    "tree_path": tree_path,
                },
            },
        ]

        dummy_vectors = [[0.01] * 768, [0.09] * 768]
        count = self.db_client.upsert_chunks_batch(sample_chunks, dummy_vectors)
        self.assertEqual(count, 2)

        # 3. Vector Similarity Search
        results = self.db_client.search_similar(query_vector=[0.01] * 768, top_k=2)
        self.assertGreaterEqual(len(results), 1)

        first_match = results[0]
        self.assertEqual(first_match["doc_name"], doc_name)
        self.assertEqual(first_match["tree_path"], tree_path)
        self.assertIn("similarity", first_match)
        self.assertGreater(first_match["similarity"], 0.9)  # Exact match vector should be ~1.0

        # Filtered search by doc_name
        filtered_results = self.db_client.search_similar(
            query_vector=[0.01] * 768, top_k=1, doc_name=doc_name
        )
        self.assertEqual(len(filtered_results), 1)
        self.assertEqual(filtered_results[0]["chunk_id"], f"{doc_id}_chunk_0")

    def test_document_indexer_file_workflow(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            chunks_path = tmp_path / "sample_chunks.json"
            tree_path = tmp_path / "sample_tree.json"

            tree_path.write_text(json.dumps({"node_type": "DocumentRoot", "text": "sample.pdf"}))
            sample_chunks = [
                {
                    "chunk_id": "sample_chunk_1",
                    "chunk_index": 0,
                    "text": "Sample text content for indexer test.",
                    "embed_text": "[sample.pdf > Sec 1]\nSample text content for indexer test.",
                    "metadata": {
                        "doc_name": "sample.pdf",
                        "page": 1,
                        "node_id": "node_1",
                        "parent_title": "Sec 1",
                        "breadcrumb": "sample.pdf > Sec 1",
                        "level": 4,
                        "node_type": "Paragraph",
                    },
                }
            ]
            chunks_path.write_text(json.dumps(sample_chunks))

            indexer = DocumentIndexer(
                embedder=self.mock_embedder,
                db_client=self.db_client,
                batch_size=10,
            )

            total = indexer.index_document_files(chunks_path, tree_path)
            self.assertEqual(total, 1)

    def test_live_ollama_nomic_embed_and_search(self):
        """End-to-end integration test with real Ollama nomic-embed-text and pgvector."""
        live_embedder = OllamaEmbedder(model_name="nomic-embed-text")
        
        doc_id = "live_test_doc"
        doc_name = "live_paper.pdf"
        tree_path = "/path/to/live_tree.json"

        chunks = [
            {
                "chunk_id": "live_chunk_attention",
                "chunk_index": 0,
                "text": "The attention mechanism allows neural networks to focus on specific parts of the input sequence.",
                "embed_text": f"[{doc_name} > Attention Mechanism]\nThe attention mechanism allows neural networks to focus on specific parts of the input sequence.",
                "metadata": {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page": 3,
                    "node_id": "node_attn",
                    "parent_title": "Attention Mechanism",
                    "breadcrumb": f"{doc_name} > Attention Mechanism",
                    "level": 4,
                    "node_type": "Paragraph",
                    "tree_path": tree_path,
                },
            },
            {
                "chunk_id": "live_chunk_optimizer",
                "chunk_index": 1,
                "text": "Stochastic gradient descent with momentum and Adam optimizer were used for training.",
                "embed_text": f"[{doc_name} > Optimization]\nStochastic gradient descent with momentum and Adam optimizer were used for training.",
                "metadata": {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page": 4,
                    "node_id": "node_opt",
                    "parent_title": "Optimization",
                    "breadcrumb": f"{doc_name} > Optimization",
                    "level": 4,
                    "node_type": "Paragraph",
                    "tree_path": tree_path,
                },
            },
        ]

        # 1. Generate real embeddings with nomic-embed-text
        texts = [str(c["embed_text"]) for c in chunks]
        vectors = live_embedder.embed_batch(texts)
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 768)

        # 2. Store in pgvector
        self.db_client.upsert_document(doc_id, doc_name, tree_path, total_chunks=2)
        count = self.db_client.upsert_chunks_batch(chunks, vectors)
        self.assertEqual(count, 2)

        # 3. Query with a semantic question using live embedding
        query_text = "How does transformer attention work?"
        query_vec = live_embedder.embed_text(query_text)
        self.assertEqual(len(query_vec), 768)

        search_results = self.db_client.search_similar(query_vector=query_vec, top_k=2)
        self.assertGreaterEqual(len(search_results), 1)

        # The top result must semantically match the attention chunk!
        top_match = search_results[0]
        self.assertEqual(top_match["chunk_id"], "live_chunk_attention")
        self.assertEqual(top_match["tree_path"], tree_path)
        self.assertGreater(top_match["similarity"], 0.6)


if __name__ == "__main__":
    unittest.main()
