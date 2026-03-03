# src/build_index.py
from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Tuple

import numpy as np

from config import Config
from ollama_client import OllamaClient
from textify import pack_to_text

def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

def load_packs(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("dashboard_chat_data.json must be a LIST of chat_context records.")
    return data

def batched(iterable: List[str], batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield i, iterable[i:i+batch_size]

def build_embeddings(
    client: OllamaClient,
    texts: List[str],
    cfg: Config
) -> np.ndarray:
    chunks = []
    for start, batch in batched(texts, cfg.batch_size):
        E = client.embed(cfg.embed_model, batch, timeout_s=cfg.request_timeout_s)
        chunks.append(E)
        print(f"Embedded {min(start+cfg.batch_size, len(texts))}/{len(texts)}")
    X = np.vstack(chunks).astype(np.float32)
    return l2_normalize(X)

def try_build_faiss_index(X: np.ndarray, out_path: str) -> bool:
    try:
        import faiss
    except Exception as e:
        print(f"[warn] FAISS not available, skipping faiss.index. ({e})")
        return False

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity with normalized vectors
    index.add(X)
    faiss.write_index(index, out_path)
    return True

def main():
    cfg = Config()
    os.makedirs(cfg.index_dir, exist_ok=True)

    packs = load_packs(cfg.data_path)
    print("Loaded records:", len(packs))

    ids = [str(p.get("note_id")) for p in packs]
    docs = [pack_to_text(p) for p in packs]

    client = OllamaClient(cfg.ollama_url)
    X = build_embeddings(client, docs, cfg)

    print("Embeddings shape:", X.shape)

    # Persist artifacts
    np.save(cfg.emb_path, X)
    with open(cfg.docs_path, "w") as f:
        json.dump(docs, f, indent=2)
    with open(cfg.ids_path, "w") as f:
        json.dump(ids, f, indent=2)

    built = try_build_faiss_index(X, cfg.faiss_path)
    if built:
        print("Saved FAISS index:", cfg.faiss_path)

    print("Saved:")
    print(" -", cfg.emb_path)
    print(" -", cfg.docs_path)
    print(" -", cfg.ids_path)

if __name__ == "__main__":
    main()