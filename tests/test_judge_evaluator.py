import json
import unittest
from unittest.mock import MagicMock, patch

from src.retrieval.evaluator import (
    CitationAudit,
    EvaluationResult,
    RAGJudge,
    audit_citations,
    build_judge_prompt,
)
from src.retrieval.orchestrator import (
    RetrievalConfig,
    RetrievalOrchestrator,
    RetrievalResult,
)


class TestCitationAuditor(unittest.TestCase):

    def test_valid_citations(self):
        answer = "FPGA accelerator latency is 12ms [Source 1, Page 3] and throughput is 100GB/s [Source 2]."
        audit = audit_citations(answer, num_sources=2)

        self.assertTrue(audit.is_valid)
        self.assertEqual(audit.total_citations, 2)
        self.assertEqual(audit.valid_citations, 2)
        self.assertEqual(audit.invalid_citations, [])
        self.assertEqual(audit.cited_source_indices, [1, 2])
        self.assertEqual(audit.uncited_source_indices, [])
        self.assertGreater(audit.citation_density_per_100_words, 0.0)

    def test_invalid_hallucinated_source_index(self):
        answer = "The system achieved 95% accuracy [Source 5]."
        # Only 2 sources were retrieved, so Source 5 is a hallucination
        audit = audit_citations(answer, num_sources=2)

        self.assertFalse(audit.is_valid)
        self.assertEqual(audit.total_citations, 1)
        self.assertEqual(audit.valid_citations, 0)
        self.assertIn("[Source 5]", audit.invalid_citations)

    def test_missing_citations_when_sources_exist(self):
        answer = "This statement does not cite anything at all."
        audit = audit_citations(answer, num_sources=3)

        self.assertFalse(audit.is_valid)
        self.assertEqual(audit.total_citations, 0)
        self.assertEqual(audit.uncited_source_indices, [1, 2, 3])


class TestRAGJudge(unittest.TestCase):

    def setUp(self):
        self.judge = RAGJudge(default_judge_model="qwen2.5:latest")

    def test_same_model_bias_warning(self):
        """When judge model equals synthesis model, same_model_bias_warning must be True."""
        with patch("src.retrieval.evaluator.ollama.chat") as mock_chat:
            mock_chat.return_value = {
                "message": {
                    "content": json.dumps({
                        "faithfulness": {"score": 0.9, "reasoning": "Well supported."},
                        "answer_relevance": {"score": 0.85, "reasoning": "Answers question."},
                        "conflict_handling": {"score": 1.0, "reasoning": "No conflict."},
                    })
                }
            }
            res = self.judge.evaluate(
                query="What is FPGA?",
                formatted_context="[Source 1] FPGA details.",
                candidate_answer="FPGA details [Source 1].",
                num_sources=1,
                synthesis_model="llama3.2:latest",
                judge_model="llama3.2:latest",  # Identical model -> should trigger warning
            )
            self.assertTrue(res.same_model_bias_warning)
            self.assertEqual(res.judge_model, "llama3.2:latest")

    def test_disjoint_model_no_bias_warning(self):
        """When judge model differs from synthesis model, warning is False."""
        with patch("src.retrieval.evaluator.ollama.chat") as mock_chat:
            mock_chat.return_value = {
                "message": {
                    "content": json.dumps({
                        "faithfulness": {"score": 0.95, "reasoning": "Extremely faithful."},
                        "answer_relevance": {"score": 0.90, "reasoning": "Direct answer."},
                        "conflict_handling": {"score": 1.0, "reasoning": "Clear."},
                    })
                }
            }
            res = self.judge.evaluate(
                query="What is FPGA?",
                formatted_context="[Source 1] FPGA details.",
                candidate_answer="FPGA details [Source 1].",
                num_sources=1,
                synthesis_model="llama3.2:latest",
                judge_model="qwen2.5:latest",  # Heterogeneous disjoint judge
            )
            self.assertFalse(res.same_model_bias_warning)
            self.assertTrue(res.passed)
            self.assertGreaterEqual(res.overall_score, 0.85)

    def test_json_extraction_with_markdown_fences(self):
        """Judge extracts JSON cleanly even when enclosed in ```json ... ``` codeblocks."""
        raw_output = """Here is the evaluation:
```json
{
  "faithfulness": {"score": 0.92, "reasoning": "Accurate"},
  "answer_relevance": {"score": 0.88, "reasoning": "Relevant"},
  "conflict_handling": {"score": 0.95, "reasoning": "Noted discrepancy"}
}
```
"""
        extracted = RAGJudge._extract_json(raw_output)
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted["faithfulness"]["score"], 0.92)

    def test_build_judge_prompt_anonymity(self):
        """Candidate answer is isolated in <candidate_answer> tags without model names."""
        prompt = build_judge_prompt("query", "context", "candidate answer")
        self.assertIn("<reference_context>", prompt)
        self.assertIn("<user_query>", prompt)
        self.assertIn("<candidate_answer>\ncandidate answer\n</candidate_answer>", prompt)
        self.assertNotIn("llama", prompt.lower())


class TestOrchestratorEvaluationIntegration(unittest.TestCase):

    @patch("src.retrieval.orchestrator.ollama.chat")
    def test_answer_with_evaluation_enabled(self, mock_chat):
        # 1st call is for answer synthesis, 2nd call is for judge evaluation
        synth_response = MagicMock()
        synth_response.message.content = "Navier-Stokes accelerated on FPGA [Source 1]."

        judge_response = {
            "message": {
                "content": json.dumps({
                    "faithfulness": {"score": 0.95, "reasoning": "Direct match."},
                    "answer_relevance": {"score": 0.90, "reasoning": "Addresses question."},
                    "conflict_handling": {"score": 1.0, "reasoning": "No conflict."},
                })
            }
        }

        mock_chat.side_effect = [synth_response, judge_response]

        mock_stage1 = MagicMock()
        mock_stage1.search.return_value = [{"chunk_id": "c1"}]
        mock_stage2 = MagicMock()
        mock_stage2.rerank.return_value = [
            {"chunk_id": "c1", "doc_name": "paper.pdf", "page": 1, "text": "FPGA simulation details.", "rerank_score": 0.9}
        ]

        orch = RetrievalOrchestrator(
            hybrid_retriever=mock_stage1,
            reranker=mock_stage2,
            db_client=MagicMock(),
        )

        res = orch.answer(
            query="Explain FPGA simulation",
            evaluate=True,
            judge_model="qwen2.5:latest",
        )

        self.assertIn("answer", res)
        self.assertIn("evaluation", res)
        eval_report: EvaluationResult = res["evaluation"]
        self.assertTrue(eval_report.passed)
        self.assertEqual(eval_report.judge_model, "qwen2.5:latest")
        self.assertFalse(eval_report.same_model_bias_warning)
        self.assertEqual(eval_report.citation_audit.valid_citations, 1)


if __name__ == "__main__":
    unittest.main()
