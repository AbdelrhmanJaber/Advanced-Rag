import unittest
from unittest.mock import MagicMock

from src.indexing.db_client import PostgresVectorClient
from src.indexing.embedder import OllamaEmbedder
from src.retrieval.hybrid_retriever import HybridRetriever, build_keyword_query

DIM = 768
TEST_DOC_ID = "hybrid_search_test_doc"
TEST_DOC_NAME = "hybrid_search_test_doc.pdf"


def _unit(*hot: int) -> list:
    v = [0.0] * DIM
    for i in hot:
        v[i] = 1.0
    return v


def _chunk(idx: int, text: str, title: str) -> dict:
    return {
        "chunk_id": f"{TEST_DOC_ID}_chunk_{idx}",
        "chunk_index": idx,
        "text": text,
        "embed_text": text,
        "metadata": {
            "doc_id": TEST_DOC_ID,
            "doc_name": TEST_DOC_NAME,
            "page": 1,
            "node_id": f"n_{idx}",
            "parent_title": title,
            "breadcrumb": f"{TEST_DOC_NAME} > {title}",
            "level": 4,
            "node_type": "Paragraph",
            "tree_path": "/tmp/tree.json",
        },
    }


class TestBuildKeywordQuery(unittest.TestCase):

    def test_strips_conversational_filler(self):
        self.assertEqual(
            build_keyword_query("Can you please explain transformer latency?"),
            "you transformer latency",
        )

    def test_keeps_identifiers(self):
        q = build_keyword_query("tell me about arXiv:2407.01528v3 and GPT-4")
        self.assertIn("2407.01528v3", q)
        self.assertIn("GPT-4", q)

    def test_falls_back_when_everything_stripped(self):
        self.assertEqual(build_keyword_query("explain"), "explain")


class TestHybridSearchDB(unittest.TestCase):
    """
    Integration test against the live pgvector DB.
    Only creates/deletes its own document, so indexed real data is left untouched.
    """

    def setUp(self):
        self.db = PostgresVectorClient()
        self._cleanup()
        self.db.upsert_document(TEST_DOC_ID, TEST_DOC_NAME, "/tmp/tree.json", total_chunks=3)
        chunks = [
            _chunk(0, "Transformer latency benchmark across batch sizes.", "Benchmarks"),
            _chunk(1, "Growing tomatoes requires sunlight and water.", "Gardening"),
            _chunk(2, "Unrelated closing remarks and acknowledgements.", "Closing"),
        ]
        # Query vector will be e0. Dense ranking: chunk_1 (sim 1.0) > chunk_0 (~0.71) > chunk_2 (0).
        vectors = [_unit(0, 1), _unit(0), _unit(2)]
        self.db.upsert_chunks_batch(chunks, vectors)

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE doc_id = %s", (TEST_DOC_ID,))
            conn.commit()

    def test_rrf_fuses_dense_and_sparse(self):
        results = self.db.hybrid_search(
            query_text="transformer latency",
            query_vector=_unit(0),
            top_k=3,
            candidate_pool=10,
            doc_name=TEST_DOC_NAME,
        )
        ids = [r["chunk_id"] for r in results]
        by_id = {r["chunk_id"]: r for r in results}

        self.assertEqual(len(results), 3)
        self.assertTrue(all(r["doc_name"] == TEST_DOC_NAME for r in results))

        # Dense side alone would put chunk_1 first; keyword side alone only finds chunk_0.
        self.assertEqual(by_id[f"{TEST_DOC_ID}_chunk_1"]["dense_rank"], 1)
        self.assertEqual(by_id[f"{TEST_DOC_ID}_chunk_0"]["sparse_rank"], 1)
        self.assertIsNone(by_id[f"{TEST_DOC_ID}_chunk_2"]["sparse_rank"])

        # Fusion: chunk_0 appears in both lists, so it wins overall.
        self.assertEqual(ids[0], f"{TEST_DOC_ID}_chunk_0")
        self.assertEqual(by_id[ids[0]]["source"], "both")
        self.assertEqual(ids, sorted(ids, key=lambda i: -by_id[i]["rrf_score"]))

    def test_keyword_side_uses_or_semantics(self):
        # Only "transformer" exists in the corpus; "zebra" does not. With AND semantics this returns nothing.
        results = self.db.hybrid_search(
            query_text="transformer zebra",
            query_vector=_unit(5),
            top_k=3,
            candidate_pool=10,
            doc_name=TEST_DOC_NAME,
        )
        sparse_hits = [r["chunk_id"] for r in results if r["sparse_rank"] is not None]
        self.assertEqual(sparse_hits, [f"{TEST_DOC_ID}_chunk_0"])

    def test_retriever_embeds_query_with_embedder(self):
        embedder = MagicMock(spec=OllamaEmbedder)
        embedder.embed_text.return_value = _unit(0)
        retriever = HybridRetriever(embedder=embedder, db_client=self.db)

        results = retriever.search("Please explain transformer latency", top_k=2, doc_name=TEST_DOC_NAME)

        embedder.embed_text.assert_called_once_with("Please explain transformer latency")
        self.assertEqual(results[0]["chunk_id"], f"{TEST_DOC_ID}_chunk_0")


if __name__ == "__main__":
    unittest.main()
