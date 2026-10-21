import unittest
from unittest.mock import MagicMock, patch

from src.retrieval.prompts import (
    DEFAULT_RAG_SYSTEM_PROMPT,
    build_rag_user_prompt,
    build_rag_prompt_messages,
)
from src.retrieval.orchestrator import (
    RetrievalConfig,
    RetrievalOrchestrator,
    RetrievalResult,
)


class TestPromptEngineeringAndConflictRules(unittest.TestCase):

    def test_system_prompt_grounding_rules(self):
        """Verify the system prompt enforces zero-hallucination and refusal on missing info."""
        prompt = DEFAULT_RAG_SYSTEM_PROMPT
        self.assertIn("GROUNDING & FACTUAL INTEGRITY", prompt)
        self.assertIn("ZERO HALLUCINATION", prompt)
        self.assertIn("insufficient information to answer", prompt)
        self.assertIn("Do NOT extrapolate", prompt)

    def test_system_prompt_citation_requirements(self):
        """Verify the system prompt requires strict inline citations."""
        prompt = DEFAULT_RAG_SYSTEM_PROMPT
        self.assertIn("STRICT INLINE CITATION REQUIREMENTS", prompt)
        self.assertIn("[Source X, Page Y]", prompt)
        self.assertIn("inline citation", prompt)
        self.assertIn("Never cite a source for information it does not contain", prompt)

    def test_system_prompt_conflict_handling_rules(self):
        """Verify the system prompt provides explicit conflict identification and attribution."""
        prompt = DEFAULT_RAG_SYSTEM_PROMPT
        self.assertIn("CONFLICT-HANDLING & DISCREPANCY RESOLUTION", prompt)
        self.assertIn("DO NOT silently pick one source", prompt)
        self.assertIn("EXPLICITLY highlight the conflict", prompt)
        self.assertIn("Relevance Rank", prompt)

    def test_system_prompt_chunk_interpretation_rules(self):
        """Verify chunk interpretation explains Lost in the Middle and metadata."""
        prompt = DEFAULT_RAG_SYSTEM_PROMPT
        self.assertIn("CHUNK INTERPRETATION", prompt)
        self.assertIn("Lost in the Middle", prompt)
        self.assertIn("Relevance Rank", prompt)

    def test_build_rag_user_prompt_delimiters(self):
        """Verify user prompt safely encapsulates context and instructions in XML tags."""
        context = "[Source 1] Page: 2\nDocument facts here."
        query = "What is the result?"
        user_prompt = build_rag_user_prompt(query, context)

        self.assertIn("<context>", user_prompt)
        self.assertIn(context, user_prompt)
        self.assertIn("</context>", user_prompt)
        self.assertIn("<instructions>", user_prompt)
        self.assertIn("<user_query>", user_prompt)
        self.assertIn(query, user_prompt)
        self.assertIn("</user_query>", user_prompt)

    def test_build_rag_prompt_messages(self):
        """Verify build_rag_prompt_messages produces valid system and user roles."""
        messages = build_rag_prompt_messages("test query", "test context")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], DEFAULT_RAG_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("<user_query>\ntest query\n</user_query>", messages[1]["content"])

    def test_orchestrator_build_prompt(self):
        """Verify orchestrator.build_prompt works with string, list of chunks, or RetrievalResult."""
        orch = RetrievalOrchestrator(
            hybrid_retriever=MagicMock(),
            reranker=MagicMock(),
            db_client=MagicMock(),
        )

        # 1. With RetrievalResult
        res = RetrievalResult(
            query="q",
            chunks=[],
            formatted_context="Sample formatted context",
            metrics={},
        )
        messages_res = orch.build_prompt("q", res)
        self.assertEqual(messages_res[0]["role"], "system")
        self.assertIn("Sample formatted context", messages_res[1]["content"])

        # 2. With raw chunks
        chunks = [
            {"doc_name": "a.pdf", "page": 1, "text": "hello", "relevance_rank": 1}
        ]
        messages_chunks = orch.build_prompt("q", chunks)
        self.assertIn("Document: a.pdf", messages_chunks[1]["content"])

    @patch("src.retrieval.orchestrator.ollama.chat")
    def test_orchestrator_answer_uses_grounded_prompts(self, mock_chat):
        """Verify orchestrator.answer constructs grounded prompt and passes to ollama."""
        mock_response = MagicMock()
        mock_response.message.content = "According to [Source 1, Page 1], the answer is X."
        mock_chat.return_value = mock_response

        mock_stage1 = MagicMock()
        mock_stage1.search.return_value = [{"chunk_id": "c1"}]
        mock_stage2 = MagicMock()
        mock_stage2.rerank.return_value = [
            {
                "chunk_id": "c1",
                "doc_name": "guide.pdf",
                "page": 2,
                "text": "Guideline details.",
                "rerank_score": 0.95,
            }
        ]

        orch = RetrievalOrchestrator(
            hybrid_retriever=mock_stage1,
            reranker=mock_stage2,
            db_client=MagicMock(),
        )

        result = orch.answer("How does it work?")

        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args[1]
        sent_messages = call_kwargs["messages"]

        # Ensure system prompt has the strict grounding and conflict rules
        self.assertEqual(sent_messages[0]["role"], "system")
        self.assertIn("GROUNDING & FACTUAL INTEGRITY", sent_messages[0]["content"])
        self.assertIn("CONFLICT-HANDLING", sent_messages[0]["content"])

        # Ensure user message has isolated context and query
        self.assertEqual(sent_messages[1]["role"], "user")
        self.assertIn("<context>", sent_messages[1]["content"])
        self.assertIn("How does it work?", sent_messages[1]["content"])

        self.assertEqual(result["answer"], "According to [Source 1, Page 1], the answer is X.")


if __name__ == "__main__":
    unittest.main()
