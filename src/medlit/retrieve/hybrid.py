import re
import json
import pickle
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Paths
PROCESSED_DATA_DIR = Path("data/processed")
CHUNKS_FILE = PROCESSED_DATA_DIR / "parkinsons_chunks.jsonl"
BM25_INDEX_FILE = PROCESSED_DATA_DIR / "parkinsons_bm25.pkl"
QDRANT_DB_PATH = Path("data/qdrant")

COLLECTION_NAME = "parkinsons_fulltext"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RRF_K = 60
TOP_K_RETRIEVE = 50


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(self):
        print("Initializing Hybrid Retriever v2 (Parkinson's Full-Text)...")

        # 1. Load matching embedding model
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

        # 2. Connect to Qdrant
        self.qdrant = QdrantClient(path=str(QDRANT_DB_PATH))

        # 3. Load or build Parkinson's BM25 index
        if BM25_INDEX_FILE.exists():
            with open(BM25_INDEX_FILE, "rb") as f:
                bm25_data = pickle.load(f)
                self.bm25 = bm25_data["bm25"]
                self.chunk_mapping = bm25_data["mapping"]
        else:
            self.bm25, self.chunk_mapping = self._build_bm25_index()

    def _build_bm25_index(self):
        """Builds and caches the BM25 index from parkinsons_chunks.jsonl."""
        print(f"Building BM25 index from {CHUNKS_FILE}...")
        chunk_mapping = []
        tokenized_corpus = []

        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                chunk_mapping.append(chunk)
                tokenized_corpus.append(tokenize(chunk["text"]))

        bm25 = BM25Okapi(tokenized_corpus)
        with open(BM25_INDEX_FILE, "wb") as f:
            pickle.dump({"bm25": bm25, "mapping": chunk_mapping}, f)

        print(f"Cached BM25 index ({len(chunk_mapping)} chunks) -> {BM25_INDEX_FILE}")
        return bm25, chunk_mapping

    def embed_query(self, query: str) -> list[float]:
        """Embeds the user question using BAAI/bge-small-en-v1.5."""
        return self.model.encode(query, normalize_embeddings=True).tolist()

    def dense_search(self, query_vector: list[float], limit: int = TOP_K_RETRIEVE):
        """Searches Qdrant using the query_points API."""
        search_result = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
        ).points

        return [
            {"chunk_id": hit.payload["chunk_id"], "payload": hit.payload}
            for hit in search_result
        ]

    def sparse_search(self, query: str, limit: int = TOP_K_RETRIEVE):
        """Searches BM25 over the Parkinson's full-text chunks."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_n_indices = scores.argsort()[-limit:][::-1]

        results = []
        for idx in top_n_indices:
            if scores[idx] > 0:
                chunk_data = self.chunk_mapping[idx]
                results.append({"chunk_id": chunk_data["chunk_id"], "payload": chunk_data})
        return results

    def retrieve(self, query: str):
        """Performs hybrid search and fuses with Reciprocal Rank Fusion (RRF)."""
        query_vector = self.embed_query(query)
        dense_results = self.dense_search(query_vector)
        sparse_results = self.sparse_search(query)

        rrf_scores = {}
        payloads = {}

        for rank, item in enumerate(dense_results):
            cid = item["chunk_id"]
            payloads[cid] = item["payload"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        for rank, item in enumerate(sparse_results):
            cid = item["chunk_id"]
            payloads[cid] = item["payload"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for cid, score in fused:
            doc = dict(payloads[cid])
            doc["rrf_score"] = score
            final_results.append(doc)

        return final_results


def main():
    retriever = HybridRetriever()
    test_query = "How are oligodendrocytes related to Parkinson's disease?"
    print(f"\nSearching for: '{test_query}'\n")

    results = retriever.retrieve(test_query)

    print(f"Found {len(results)} fused chunks. Top 3:")
    for i, res in enumerate(results[:3]):
        print(f"\n[{i + 1}] PMID: {res['pmid']} | Section: {res.get('section', 'N/A')} (RRF: {res['rrf_score']:.4f})")
        print(f"Title: {res['title']}")
        print(f"Snippet: {res['text'][:180]}...")


if __name__ == "__main__":
    main()