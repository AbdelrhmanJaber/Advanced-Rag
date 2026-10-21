import multiprocessing as mp
import unittest
from unittest.mock import MagicMock, patch

from src.rag_orchestrator import (
    MasterRAGOrchestrator,
    RAGPipelineConfig,
    SENTINEL,
    lock_and_release,
)
from src.retrieval.orchestrator import RetrievalResult


class TestMasterRAGOrchestrator(unittest.TestCase):

    def test_default_config(self):
        orch = MasterRAGOrchestrator()
        cfg = orch.config
        self.assertEqual(cfg.n_validator_workers, 4)
        self.assertEqual(cfg.n_classifier_workers, 4)
        self.assertEqual(cfg.n_native_parser_workers, 4)
        self.assertEqual(cfg.n_scanned_parser_workers, 2)
        self.assertEqual(cfg.n_indexer_workers, 2)
        self.assertEqual(cfg.synthesis_model, "llama3.2:latest")
        self.assertEqual(cfg.judge_model, "qwen2.5:latest")
        self.assertTrue(cfg.reorder_lost_in_middle)

    def test_lock_and_release_decrements_and_signals(self):
        counter = mp.Value("i", 2)
        lock = mp.Lock()
        q_downstream: mp.Queue = mp.Queue()

        # 1st worker finishes -> counter becomes 1, no sentinels sent
        lock_and_release(counter, lock, [(q_downstream, 3)])
        self.assertEqual(counter.value, 1)
        self.assertTrue(q_downstream.empty())

        # 2nd worker finishes -> counter becomes 0, sends 3 sentinels
        lock_and_release(counter, lock, [(q_downstream, 3)])
        self.assertEqual(counter.value, 0)

        sentinels = [q_downstream.get(timeout=1.0) for _ in range(3)]
        self.assertEqual(sentinels, [SENTINEL, SENTINEL, SENTINEL])

    @patch("src.rag_orchestrator.RetrievalOrchestrator.answer")
    def test_ask_delegation(self, mock_answer):
        fake_result = {
            "answer": "Test answer [Source 1].",
            "retrieval": RetrievalResult(query="test", chunks=[], formatted_context=""),
        }
        mock_answer.return_value = fake_result

        orch = MasterRAGOrchestrator()
        result = orch.ask(
            query="test question",
            evaluate=True,
            synthesis_model="llama3.2:latest",
            judge_model="qwen2.5:latest",
            top_k=3,
        )

        self.assertEqual(result["answer"], "Test answer [Source 1].")
        mock_answer.assert_called_once()
        call_kwargs = mock_answer.call_args[1]
        self.assertEqual(call_kwargs["query"], "test question")
        self.assertEqual(call_kwargs["model"], "llama3.2:latest")
        self.assertEqual(call_kwargs["evaluate"], True)
        self.assertEqual(call_kwargs["judge_model"], "qwen2.5:latest")


if __name__ == "__main__":
    unittest.main()
