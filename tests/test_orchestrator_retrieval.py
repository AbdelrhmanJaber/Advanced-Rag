import unittest
from unittest.mock import MagicMock, patch

from src.indexing.db_client import PostgresVectorClient
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import MxbaiReranker
from src.retrieval.orchestrator import (
    RetrievalConfig,
    RetrievalResult,
    RetrievalOrchestrator,
    reorder_lost_in_middle,
)


class TestRetrievalOrchestrator(unittest.TestCase):

    def setUp(self):
        self.mock_stage1 = MagicMock(spec=HybridRetriever)
        self.mock_stage2 = MagicMock(spec=MxbaiReranker)
        self.mock_db = MagicMock(spec=PostgresVectorClient)
        self.orchestrator = RetrievalOrchestrator(
            hybrid_retriever=self.mock_stage1,
            reranker=self.mock_stage2,
            db_client=self.mock_db,
        )

    def test_empty_query_returns_empty_result(self):
        res = self.orchestrator.retrieve("")
        self.assertEqual(res.chunks, [])
        self.assertEqual(res.formatted_context, "")
        self.assertEqual(res.metrics["total_time_ms"], 0.0)

        res_spaces = self.orchestrator.retrieve("   ")
        self.assertEqual(res_spaces.chunks, [])

    def test_retrieve_end_to_end(self):
        self.mock_stage1.search.return_value = [
            {"chunk_id": "c1", "doc_name": "doc1.pdf", "page": 1, "text": "Text 1", "rrf_score": 0.02},
            {"chunk_id": "c2", "doc_name": "doc1.pdf", "page": 2, "text": "Text 2", "rrf_score": 0.01},
        ]
        self.mock_stage2.rerank.return_value = [
            {"chunk_id": "c2", "doc_name": "doc1.pdf", "page": 2, "text": "Text 2", "rerank_score": 0.88},
            {"chunk_id": "c1", "doc_name": "doc1.pdf", "page": 1, "text": "Text 1", "rerank_score": 0.65},
        ]

        cfg = RetrievalConfig(top_k=2, candidate_pool=10)
        res = self.orchestrator.retrieve("test question", config=cfg)

        self.assertEqual(len(res.chunks), 2)
        self.assertEqual(res.chunks[0]["chunk_id"], "c2")
        self.assertIn("[Source 1]", res.formatted_context)
        self.assertIn("Text 2", res.formatted_context)
        self.assertIn("total_time_ms", res.metrics)
        self.assertEqual(res.metrics["candidates_retrieved"], 2)

    def test_min_rerank_score_filtering(self):
        self.mock_stage1.search.return_value = [{"chunk_id": "c1"}]
        self.mock_stage2.rerank.return_value = [
            {"chunk_id": "c1", "text": "High score", "rerank_score": 0.85},
            {"chunk_id": "c2", "text": "Low score", "rerank_score": 0.35},
        ]

        cfg = RetrievalConfig(top_k=5, min_rerank_score=0.70)
        res = self.orchestrator.retrieve("filter test", config=cfg)

        self.assertEqual(len(res.chunks), 1)
        self.assertEqual(res.chunks[0]["chunk_id"], "c1")

    def test_format_context_with_tables_and_breadcrumbs(self):
        chunks = [
            {
                "doc_name": "benchmark.pdf",
                "page": 3,
                "breadcrumb": "benchmark.pdf > Results",
                "rerank_score": 0.92,
                "text": "The table below lists hardware acceleration latency.",
                "table_markdown": "| Model | Latency |\n|---|---|\n| FPGA | 12ms |",
            }
        ]

        formatted = self.orchestrator.format_context(chunks)
        self.assertIn("[Source 1] Document: benchmark.pdf | Page: 3 | Section: benchmark.pdf > Results", formatted)
        self.assertIn("Table Data:\n| Model | Latency |", formatted)

    def test_context_window_expansion(self):
        self.mock_stage1.search.return_value = [{"chunk_id": "c1"}]
        self.mock_stage2.rerank.return_value = [
            {
                "chunk_id": "c_5",
                "doc_name": "paper.pdf",
                "chunk_index": 5,
                "text": "Target middle chunk",
                "rerank_score": 0.85,
            }
        ]
        self.mock_db.get_surrounding_chunks.return_value = [
            {"chunk_id": "c_4", "text": "Preceding chunk text"},
            {"chunk_id": "c_5", "text": "Target middle chunk"},
            {"chunk_id": "c_6", "text": "Following chunk text"},
        ]

        cfg = RetrievalConfig(top_k=1, expand_window=1)
        res = self.orchestrator.retrieve("expansion test", config=cfg)

        self.assertEqual(len(res.chunks), 1)
        chunk = res.chunks[0]
        self.assertIn("Preceding chunk text", chunk["expanded_text"])
        self.assertIn("Following chunk text", chunk["expanded_text"])
        self.mock_db.get_surrounding_chunks.assert_called_once_with(
            doc_name="paper.pdf",
            chunk_index=5,
            window=1,
        )

    @patch("src.retrieval.orchestrator.ollama.chat")
    def test_answer_generation(self, mock_chat):
        mock_response = MagicMock()
        mock_response.message.content = "According to [Source 1], random attention models..."
        mock_chat.return_value = mock_response

        self.mock_stage1.search.return_value = [{"chunk_id": "c1"}]
        self.mock_stage2.rerank.return_value = [
            {
                "chunk_id": "c1",
                "doc_name": "doc0.pdf",
                "page": 1,
                "breadcrumb": "Abstract",
                "text": "Random attention model details.",
                "rerank_score": 0.90,
            }
        ]

        result = self.orchestrator.answer("What is the model?", model="llama3.2:latest")

        self.assertIn("answer", result)
        self.assertEqual(result["answer"], "According to [Source 1], random attention models...")
        self.assertIn("llm_time_ms", result["retrieval"].metrics)
        mock_chat.assert_called_once()

    def test_reorder_lost_in_middle_logic(self):
        chunks = [
            {"id": "c1", "score": 0.95},
            {"id": "c2", "score": 0.85},
            {"id": "c3", "score": 0.75},
            {"id": "c4", "score": 0.65},
            {"id": "c5", "score": 0.55},
        ]
        reordered = reorder_lost_in_middle(chunks)
        self.assertEqual(len(reordered), 5)
        # Rank 1 at the start (primacy)
        self.assertEqual(reordered[0]["id"], "c1")
        # Rank 3 next
        self.assertEqual(reordered[1]["id"], "c3")
        # Rank 5 (lowest) buried in dead middle
        self.assertEqual(reordered[2]["id"], "c5")
        # Rank 4 next
        self.assertEqual(reordered[3]["id"], "c4")
        # Rank 2 at the end (recency)
        self.assertEqual(reordered[4]["id"], "c2")

    def test_orchestrator_reorder_lost_in_middle_e2e(self):
        self.mock_stage1.search.return_value = [{"chunk_id": f"c{i}"} for i in range(1, 6)]
        self.mock_stage2.rerank.return_value = [
            {"chunk_id": f"c{i}", "doc_name": "test.pdf", "page": 1, "text": f"Passage {i}", "rerank_score": 1.0 - (i * 0.1)}
            for i in range(1, 6)
        ]

        cfg = RetrievalConfig(top_k=5, reorder_lost_in_middle=True)
        res = self.orchestrator.retrieve("query", config=cfg)

        self.assertEqual(len(res.chunks), 5)
        self.assertTrue(res.metrics["reordered_lost_in_middle"])
        # First chunk is c1 (Rank 1)
        self.assertEqual(res.chunks[0]["chunk_id"], "c1")
        self.assertEqual(res.chunks[0]["relevance_rank"], 1)
        # Middle chunk is c5 (Rank 5)
        self.assertEqual(res.chunks[2]["chunk_id"], "c5")
        self.assertEqual(res.chunks[2]["relevance_rank"], 5)
        # Last chunk is c2 (Rank 2)
        self.assertEqual(res.chunks[4]["chunk_id"], "c2")
        self.assertEqual(res.chunks[4]["relevance_rank"], 2)

        # Context formatting retains clarity
        self.assertIn("Relevance Rank: #1", res.formatted_context)
        self.assertIn("Relevance Rank: #2", res.formatted_context)


class TestLiveRetrievalOrchestrator(unittest.TestCase):

    def test_live_retrieval_against_real_db(self):
        """Integration test querying real chunks in PostgreSQL."""
        orch = RetrievalOrchestrator()
        cfg = RetrievalConfig(top_k=3, candidate_pool=15)
        res = orch.retrieve("FPGA Navier Stokes simulation accelerator", config=cfg)

        self.assertGreaterEqual(len(res.chunks), 1)
        self.assertIn("[Source 1]", res.formatted_context)
        self.assertGreater(res.metrics["total_time_ms"], 0.0)
        self.assertIn("doc_3_2411.16245v2.pdf", res.chunks[0]["doc_name"])


if __name__ == "__main__":
    unittest.main()
