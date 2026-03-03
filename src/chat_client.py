# src/chat_client.py
from __future__ import annotations
import requests

class OllamaChatClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def generate(self, model: str, prompt: str, timeout_s: int = 120, temperature: float = 0.2) -> str:
        r = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout_s,
        )
        r.raise_for_status()
        return r.json()["response"]