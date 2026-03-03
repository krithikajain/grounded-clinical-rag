# src/ollama_client.py
from __future__ import annotations
import requests
from typing import List
import numpy as np

class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def embed(self, model: str, inputs: List[str], timeout_s: int = 300) -> np.ndarray:
        """
        Calls Ollama /api/embed with a batch of strings.
        Returns: (n, d) float32 numpy array
        """
        r = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": inputs},
            timeout=timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        embs = data.get("embeddings")
        if not embs:
            raise RuntimeError(f"Unexpected embed response keys: {list(data.keys())}")
        return np.array(embs, dtype=np.float32)