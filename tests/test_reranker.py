import unittest
from unittest.mock import MagicMock, patch

from src.retrieval.reranker import CrossEncoderReranker, MxbaiReranker, _cosine_similarity
from src.retrieval.pipeline import TwoStageRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


class TestMxbaiReranker(unittest.TestCase):

    def test_cosine_similarity(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(_cosine_similarity(v1, v2), 1.0, places=4)
        self.assertAlmostEqual(_cosine_similarity(v1, v3), 0.0, places=4)

    def test_cosine_similarity_zero_norm(self):
        # Guards against division by zero
        zero_v = [0.0, 0.0, 0.0]
        normal_v = [1.0, 2.0, 3.0]
        self.assertEqual(_cosine_similarity(zero_v, normal_v), 0.0)
        self.assertEqual(_cosine_similarity(normal_v, zero_v), 0.0)
        self.assertEqual(_cosine_similarity(zero_v, zero_v), 0.0)

    def test_empty_candidates(self):
        reranker = MxbaiReranker()
        self.assertEqual(reranker.rerank("some query", [], top_k=5), [])

    def test_non_positive_top_k(self):
        reranker = MxbaiReranker()
        candidates = [{"chunk_id": "c1", "text": "hello"}]
        self.assertEqual(reranker.rerank("some query", candidates, top_k=0), [])
        self.assertEqual(reranker.rerank("some query", candidates, top_k=-1), [])

    def test_empty_query_fallback(self):
        reranker = MxbaiReranker()
        candidates = [
            {"chunk_id": "c1", "text": "chunk one"},
            {"chunk_id": "c2", "text": "chunk two"},
        ]
        # Should not crash or make embedding calls; returns candidates[:top_k] with fallback score
        results_blank = reranker.rerank("", candidates, top_k=1)
        self.assertEqual(len(results_blank), 1)
        self.assertEqual(results_blank[0]["chunk_id"], "c1")
        self.assertEqual(results_blank[0]["rerank_score"], 0.0)

        results_spaces = reranker.rerank("   \t\n  ", candidates, top_k=2)
        self.assertEqual(len(results_spaces), 2)
        self.assertEqual(results_spaces[1]["chunk_id"], "c2")

    @patch("src.retrieval.reranker.ollama.embed")
    def test_empty_text_and_missing_field(self, mock_embed):
        query_resp = MagicMock()
        query_resp.embeddings = [[1.0, 0.0]]

        passage_resp = MagicMock()
        passage_resp.embeddings = [[0.8, 0.2], [0.1, 0.9]]

        mock_embed.side_effect = [query_resp, passage_resp]

        reranker = MxbaiReranker()
        candidates = [
            {"chunk_id": "empty_chunk", "text": ""},
            {"chunk_id": "missing_text_chunk", "breadcrumb": "Section 1"},
        ]

        results = reranker.rerank(query="Find something", candidates=candidates, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "empty_chunk")

        # Verify that empty text was normalized to "empty"
        passage_call_args = mock_embed.call_args_list[1]
        inputs = passage_call_args[1]["input"]
        self.assertIn("empty", inputs[0])

    @patch("src.retrieval.reranker.ollama.embed")
    def test_mismatched_passage_response_length(self, mock_embed):
        # Case where API returns fewer vectors than requested candidates
        query_resp = MagicMock()
        query_resp.embeddings = [[1.0, 0.0]]

        passage_resp = MagicMock()
        passage_resp.embeddings = [[0.9, 0.1]]  # Only 1 vector returned for 2 candidates

        mock_embed.side_effect = [query_resp, passage_resp]

        reranker = MxbaiReranker()
        candidates = [
            {"chunk_id": "c1", "text": "First passage"},
            {"chunk_id": "c2", "text": "Second passage"},
        ]

        results = reranker.rerank(query="Test", candidates=candidates, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0]["rerank_score"], 0.9938, places=3)
        self.assertEqual(results[1]["rerank_score"], 0.0)

    @patch("src.retrieval.reranker.ollama.embed")
    def test_rerank_ordering_mocked(self, mock_embed):
        query_resp = MagicMock()
        query_resp.embeddings = [[1.0, 0.0]]

        passage_resp = MagicMock()
        passage_resp.embeddings = [[0.2, 0.8], [0.9, 0.1]]

        mock_embed.side_effect = [query_resp, passage_resp]

        reranker = MxbaiReranker(model_name="mxbai-embed-large:latest")
        candidates = [
            {"chunk_id": "c1", "text": "Passage low similarity"},
            {"chunk_id": "c2", "text": "Passage high similarity"},
        ]

        results = reranker.rerank(query="Test query", candidates=candidates, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "c2")
        self.assertEqual(results[1]["chunk_id"], "c1")
        self.assertGreater(results[0]["rerank_score"], results[1]["rerank_score"])

    def test_live_mxbai_reranker(self):
        """End-to-end test using real Ollama mxbai-embed-large:latest model."""
        reranker = MxbaiReranker(model_name="mxbai-embed-large:latest")
        query = "How does transformer attention mechanism work?"
        candidates = [
            {
                "chunk_id": "cooking_doc",
                "text": "To bake bread, mix flour, yeast, and warm water. Let the dough rise for two hours.",
                "breadcrumb": "Cookbook > Baking",
            },
            {
                "chunk_id": "attention_doc",
                "text": "The attention mechanism in transformers allows the model to dynamically weight token representations across layers.",
                "breadcrumb": "Deep Learning > Attention Mechanism",
            },
            {
                "chunk_id": "gardening_doc",
                "text": "Tomatoes require full sunlight and well-drained soil with regular watering during dry periods.",
                "breadcrumb": "Gardening > Vegetables",
            },
        ]

        reranked = reranker.rerank(query=query, candidates=candidates, top_k=3)

        self.assertEqual(len(reranked), 3)
        # The attention chunk must be ranked #1
        self.assertEqual(reranked[0]["chunk_id"], "attention_doc")
        self.assertGreater(reranked[0]["rerank_score"], 0.6)
        self.assertGreater(reranked[0]["rerank_score"], reranked[1]["rerank_score"])
        self.assertIn("rerank_model", reranked[0])
        self.assertEqual(reranked[0]["rerank_model"], "mxbai-embed-large:latest")


class TestTwoStagePipeline(unittest.TestCase):

    def test_pipeline_orchestration(self):
        mock_stage1 = MagicMock(spec=HybridRetriever)
        mock_stage1.search.return_value = [
            {"chunk_id": "chunk_1", "text": "First candidate", "rrf_score": 0.03},
            {"chunk_id": "chunk_2", "text": "Second candidate", "rrf_score": 0.02},
        ]

        mock_stage2 = MagicMock(spec=MxbaiReranker)
        mock_stage2.rerank.return_value = [
            {"chunk_id": "chunk_2", "text": "Second candidate", "rerank_score": 0.95},
            {"chunk_id": "chunk_1", "text": "First candidate", "rerank_score": 0.40},
        ]

        pipeline = TwoStageRetriever(hybrid_retriever=mock_stage1, reranker=mock_stage2)
        results = pipeline.retrieve("Sample query", top_k=2, candidate_pool=10)

        mock_stage1.search.assert_called_once_with(
            query="Sample query",
            top_k=10,
            candidate_pool=10,
            doc_name=None,
            parent_title=None,
        )
        mock_stage2.rerank.assert_called_once()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "chunk_2")

    def test_pipeline_empty_candidates(self):
        mock_stage1 = MagicMock(spec=HybridRetriever)
        mock_stage1.search.return_value = []
        mock_stage2 = MagicMock(spec=MxbaiReranker)

        pipeline = TwoStageRetriever(hybrid_retriever=mock_stage1, reranker=mock_stage2)
        results = pipeline.retrieve("Empty search", top_k=5)
        self.assertEqual(results, [])
        mock_stage2.rerank.assert_not_called()


class TestCrossEncoderReranker(unittest.TestCase):

    def test_empty_candidates(self):
        reranker = CrossEncoderReranker()
        self.assertEqual(reranker.rerank("some query", [], top_k=5), [])

    def test_non_positive_top_k(self):
        reranker = CrossEncoderReranker()
        candidates = [{"chunk_id": "c1", "text": "hello"}]
        self.assertEqual(reranker.rerank("some query", candidates, top_k=0), [])
        self.assertEqual(reranker.rerank("some query", candidates, top_k=-1), [])

    def test_empty_query_fallback(self):
        reranker = CrossEncoderReranker()
        candidates = [
            {"chunk_id": "c1", "text": "chunk one"},
            {"chunk_id": "c2", "text": "chunk two"},
        ]
        results_blank = reranker.rerank("", candidates, top_k=1)
        self.assertEqual(len(results_blank), 1)
        self.assertEqual(results_blank[0]["chunk_id"], "c1")
        self.assertEqual(results_blank[0]["rerank_score"], 0.0)

        results_spaces = reranker.rerank("   \t\n  ", candidates, top_k=2)
        self.assertEqual(len(results_spaces), 2)
        self.assertEqual(results_spaces[1]["chunk_id"], "c2")

    def test_live_cross_encoder_rerank(self):
        reranker = CrossEncoderReranker()
        query = "How does attention mechanism work?"
        candidates = [
            {
                "chunk_id": "irrelevant_doc",
                "text": "PostgreSQL database uses GIN indexing for inverted full-text search.",
                "breadcrumb": "Database > PostgreSQL",
            },
            {
                "chunk_id": "relevant_doc",
                "text": "The attention mechanism allows neural networks to focus on specific parts of the input sequence.",
                "breadcrumb": "Deep Learning > Transformers",
            },
            {
                "chunk_id": "semi_doc",
                "text": "Neural networks are trained using backpropagation and gradient descent algorithms.",
                "breadcrumb": "Machine Learning > Optimization",
            },
        ]

        reranked = reranker.rerank(query=query, candidates=candidates, top_k=2)
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0]["chunk_id"], "relevant_doc")
        self.assertGreater(reranked[0]["rerank_score"], reranked[1]["rerank_score"])
        self.assertGreater(reranked[0]["rerank_score"], 0.8)
        self.assertEqual(reranked[0]["rerank_model"], "cross-encoder/ms-marco-MiniLM-L-6-v2")


if __name__ == "__main__":
    unittest.main()

