# src/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    ollama_url: str = "http://127.0.0.1:11434"
    embed_model: str = "nomic-embed-text"

    data_path: str = "data/dashboard_chat_data.json"

    index_dir: str = "index"
    docs_path: str = "index/docs.json"
    ids_path: str = "index/doc_ids.json"
    emb_path: str = "index/embeddings.npy"
    faiss_path: str = "index/faiss.index"

    # Performance knobs
    batch_size: int = 16
    request_timeout_s: int = 300