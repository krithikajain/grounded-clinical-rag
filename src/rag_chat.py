# src/rag_chat.py
from __future__ import annotations

import json
import numpy as np

from config import Config
from ollama_client import OllamaClient
from chat_client import OllamaChatClient

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

SYSTEM_RULES = """You are a helpful assistant answering questions about extracted clinical-note dashboard data.
    Hard rules:
    - Do not add medical advice, diagnoses, or severity claims.
    - If the answer is not explicitly present in CONTEXT, say: "I don't see that in the extracted dashboard data."
    - When you state a fact, cite the note_id(s) where you found it.
    Style:
    - Conversational, 3-6 sentences max.
    """

def build_prompt(question: str, hits: list[dict]) -> str:
    context_blocks = []
    used_ids = []
    for h in hits:
        used_ids.append(h["note_id"])
        context_blocks.append(
            f"[rank={h['rank']} score={h['score']:.3f} note_id={h['note_id']}]\n{h['doc']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    return f"""{SYSTEM_RULES}

QUESTION:
{question}

CONTEXT:
{context}

Answer (include note_id citations at the end):"""

def main():
    cfg = Config()
    embedder = OllamaClient(cfg.ollama_url)
    llm = OllamaChatClient(cfg.ollama_url)

    doc_ids = load_json(cfg.ids_path)
    docs = load_json(cfg.docs_path)

    import faiss
    index = faiss.read_index(cfg.faiss_path)

    print(f"Loaded index with {index.ntotal} vectors.")
    print("Chat with your dashboard data. Type 'exit' to quit.\n")

    while True:
        q = input("You> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        # Retrieve
        q_emb = embedder.embed(cfg.embed_model, [q], timeout_s=cfg.request_timeout_s)
        q_emb = l2_normalize(q_emb)

        k = 6
        scores, idxs = index.search(q_emb, k)

        hits = []
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
            hits.append({
                "rank": rank,
                "score": float(score),
                "note_id": doc_ids[idx],
                "doc": docs[idx],
            })

        prompt = build_prompt(q, hits)
        answer = llm.generate(
            model="gpt-oss:latest",
            prompt=prompt,
            timeout_s=180,
            temperature=0.0,
        )

        print("\nAssistant:", answer)
        print("\nRetrieved note_ids:", [h["note_id"] for h in hits])
        print("\n" + "-" * 70 + "\n")

if __name__ == "__main__":
    main()