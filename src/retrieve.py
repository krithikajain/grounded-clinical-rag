# src/retrieve.py
from __future__ import annotations

import json
import numpy as np

from config import Config
from ollama_client import OllamaClient

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def main():
    cfg = Config()
    client = OllamaClient(cfg.ollama_url)

    # Load artifacts
    doc_ids = load_json(cfg.ids_path)
    docs = load_json(cfg.docs_path)

    try:
        import faiss
    except Exception as e:
        raise RuntimeError(f"FAISS not installed/available: {e}")

    index = faiss.read_index(cfg.faiss_path)

    print(f"Loaded index with {index.ntotal} vectors.")
    print("Type a question and I'll retrieve the most relevant notes.")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Query> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        # Embed query
        q_emb = client.embed(cfg.embed_model, [q], timeout_s=cfg.request_timeout_s)
        q_emb = l2_normalize(q_emb)

        # Search
        k = 5
        scores, idxs = index.search(q_emb, k)

        print("\nTop matches:")
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
            note_id = doc_ids[idx]
            preview = docs[idx].splitlines()[:8]  # first few lines
            preview_text = "\n".join(preview)
            print(f"\n[{rank}] score={float(score):.3f}  note_id={note_id}")
            print(preview_text)

        print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    main()