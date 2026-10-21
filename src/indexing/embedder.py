from __future__ import annotations

import logging
from typing import List, Optional
import ollama

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """
    Embedding service using Ollama models (default: nomic-embed-text, 768 dimensions).
    """

    def __init__(self, model_name: str = "nomic-embed-text") -> None:
        self.model_name = model_name

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        if not text.strip():
            text = "empty"
        response = ollama.embeddings(model=self.model_name, prompt=text)
        return response["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings sequentially or in batch."""
        embeddings: List[List[float]] = []
        for text in texts:
            embeddings.append(self.embed_text(text))
        return embeddings
