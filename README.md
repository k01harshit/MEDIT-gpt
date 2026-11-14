# MedLit Copilot

An evidence-based Pharmacovigilance Retrieval-Augmented Generation (RAG) assistant designed to query and analyze PubMed medical literature.

## Architecture
* **Ingestion:** Biopython Entrez for PubMed abstract retrieval and JSONL formatting.
* **Retrieval:** Hybrid search utilizing Qdrant (dense vectors via BAAI/bge-small-en-v1.5) and BM25 (sparse indexing), unified via Reciprocal Rank Fusion (RRF).
* **Reranking:** Cross-Encoder (BAAI/bge-reranker-base) for contextual relevance scoring.
* **Generation:** LangChain integration with Groq for high-speed inference, strictly grounded in retrieved context to prevent hallucination.
* **Orchestration:** LangGraph state machine for workflow routing.
* **Interface:** Streamlit web application.

## Setup Instructions
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt` (or via conda).
3. Create a `.env` file in the root directory with `GROQ_API_KEY`.
4. Run ingestion and indexing pipelines from `src/medlit/ingest/` and `src/medlit/index/`.
5. Launch the application: `python -m streamlit run src/medlit/app.py`