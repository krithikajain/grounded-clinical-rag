# Clinical Dashboard RAG
Embedding-Based Retrieval-Augmented Generation for Structured Clinical Notes

## Overview

Clinical Dashboard RAG is an end-to-end AI system that transforms unstructured clinical notes into structured dashboard data and enables grounded question answering using a local embedding-based RAG pipeline.

This project demonstrates:

- Clinical entity extraction and contextual enrichment
- Structured dashboard JSON generation
- Embedding-based semantic retrieval using Ollama
- FAISS vector search
- Grounded response generation using a local LLM
- Clean, modular AI engineering architecture

All models run locally using Ollama.

---

## Architecture

Pipeline Overview:

Raw Clinical Notes  
→ Clinical NER + Cleaning  
→ Context Enrichment (negation, temporality, grouping)  
→ Structured Dashboard JSON  
→ Embedding (Ollama: nomic-embed-text)  
→ FAISS Vector Index  
→ Query Embedding  
→ Top-K Retrieval  
→ Grounded Generation (Ollama LLM)

This is a proper embedding-based Retrieval-Augmented Generation (RAG) system.

---

## Requirements

- Python 3.9+
- Ollama (installed locally)
- Models:
  - `nomic-embed-text`
  - `gpt-oss:latest` (or tinyllama)

---

## Setup Instructions

### 1. Clone Repository

### 2. Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

### 3. Install Dependencies
python3 -m venv .venv
source .venv/bin/activate

### Ollama Setup
### run rag_chat.py for talking with the chatbot




---

## Key Design Decisions

### 1. Embedding-Based Retrieval
We use `nomic-embed-text` via Ollama for semantic similarity search rather than keyword-based retrieval.

### 2. Cosine Similarity with FAISS
Vectors are L2-normalized and indexed using `IndexFlatIP` to approximate cosine similarity.

### 3. Grounded Generation
The LLM receives only retrieved context. It is explicitly instructed not to hallucinate or provide medical advice.

### 4. Modular Architecture
- Embedding client isolated
- Chat client isolated
- Text construction separated
- Config centralized

This makes the system extensible and production-ready.

---

## Limitations

- Small local models may occasionally hallucinate.
- Retrieval quality depends on structured extraction quality.
- Not intended for medical diagnosis or clinical decision-making.

---

## Future Improvements

- Structured field-aware retrieval
- Citation-enforced prompting
- Streaming responses
- Evaluation metrics (MRR, Recall@K)
- Web interface (Streamlit / FastAPI)

---

## Disclaimer

This system is for research and educational purposes only.
It does not provide medical advice.
